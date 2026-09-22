"""盲审分配求解器（OR-Tools CP-SAT）。

为一份**已完成且可行**的组卷结果（1~5 份卷）在同一个联合 CP-SAT 模型中
分配审阅人，硬性约束：

- 每份卷恰好分配 ``required``（每卷所需审阅人数）名审阅人，同卷审阅人不重复
  （用 0/1 变量天然保证）；
- 回避关系：审阅人声明的“不可审卷号”上该审阅人变量恒为 0；
- 总容量：每名审阅人分到的卷数不超过其个人容量，所有容量之和足以覆盖
  全部需求（否则无解）。

择优规则（字典序，逐层最优后固定再优化下一层，结果可复现）：
  1. 最大个人负载（各审阅人分到的卷数最大值）最小；
  2. 各人负载相对平均值的绝对偏差总和
     Σ_r |load_r - 总需求/人数| 最小（以人数放大为整数线性化）；
  3. 按卷号展平的审阅人编号序列最小：每份卷的审阅人编号先升序排列，再按
     卷序 0,1,... 依次拼接；审阅人按编号规范化排序（整数型按数值、其余按
     字符串）后以序号编码，序列按 R 进制分块、按 (卷序, 块序) 逐块最小化，
     由此唯一破缺同卷审阅人排列对称，同请求结果确定可复现。

无解时**绝不返回部分方案**：诊断信息逐卷给出可用人力（可审该卷的审阅人
编号与人数，已扣除回避关系）及该卷人力缺口，并给出总容量缺口
（总需求 - 容量总和，非正时为 0）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ortools.sat.python import cp_model

STAGE_TIME_LIMIT = 20.0  # 每个择优阶段的求解上限（秒）


# --------------------------------------------------------------------------- #
# 数据整理
# --------------------------------------------------------------------------- #
def _id_sort_key(rid: Any):
    """编号规范化排序：整数型编号在前按数值排，其余按字符串排。"""
    if isinstance(rid, bool):  # guard: bool 是 int 的子类
        return (1, str(rid))
    if isinstance(rid, int):
        return (0, rid)
    s = str(rid)
    if s.lstrip("-").isdigit():
        return (0, int(s))
    return (1, s)


def _read_x(
    reviewers: List[Dict[str, Any]], P: int
) -> Tuple[List[Any], List[int], List[set]]:
    """审阅人按编号规范化排序（确定性序号编码），返回 (编号, 容量, 逐卷回避集合)。"""
    ordered = sorted(reviewers, key=lambda r: _id_sort_key(r["id"]))
    ids = [r["id"] for r in ordered]
    capacities = [int(r["capacity"]) for r in ordered]
    # 仅保留有效卷序号（越界由接口层校验拦截，这里再防御一次）
    forbidden = [
        {int(p) for p in (r.get("forbidden_papers") or []) if 0 <= int(p) < P}
        for r in ordered
    ]
    return ids, capacities, forbidden


def _chunk_bits(R: int) -> int:
    """R 进制分块每块位数：块值上限 R^k 须在 int64 安全范围内（< 2^62）。"""
    bits = 62
    while bits >= 1 and R ** bits >= (1 << 62):
        bits -= 1
    return max(1, bits)


# --------------------------------------------------------------------------- #
# 建模
# --------------------------------------------------------------------------- #
def _build_model(
    R: int,
    P: int,
    required: int,
    capacities: List[int],
    forbidden: List[set],
) -> Tuple[cp_model.CpModel, Dict[str, Any]]:
    """联合分配模型。

    x[r][p]：审阅人 r 是否审第 p 卷。每卷恰好 required 人、回避位恒 0、
    每人负载不超过容量。

    order[j][p] 为“第 p 卷第 j 小（0 起）的被选审阅人序号”（j 为卷内
    升序名次，p 为卷序），经位置指示变量 pos[r][p][j] 与 x 联动，供第 3
    层“按卷号展平”的序列最小化读取：pos=1 当且仅当本人被选中且序号
    比其小者恰有 j 人。
    """
    m = cp_model.CpModel()

    x = [[m.NewBoolVar(f"x_{r}_{p}") for p in range(P)] for r in range(R)]
    loads = [sum(x[r]) for r in range(R)]

    for r in range(R):
        m.Add(loads[r] <= capacities[r])
        for p in range(P):
            if p in forbidden[r]:
                m.Add(x[r][p] == 0)

    for p in range(P):
        m.Add(sum(x[r][p] for r in range(R)) == required)

    max_load = m.NewIntVar(0, min(max(capacities), P), "max_load")
    for r in range(R):
        m.Add(max_load >= loads[r])

    # 每人负载相对平均值的绝对偏差，放大 R 倍转为整数：
    # dev_r = |R*load_r - D|，D = P*required 为总需求（平均负载 = D/R）
    D = P * required
    devs: List[cp_model.IntVar] = []
    dev_bound = max(R * max(capacities), D)
    for r in range(R):
        dev = m.NewIntVar(0, dev_bound, f"dev_{r}")
        m.Add(dev >= R * loads[r] - D)
        m.Add(dev >= D - R * loads[r])
        devs.append(dev)

    # 每卷可分配审阅人按序号升序排列（order[j][p] = 第 j 小者的序号）。
    # pos[r][p][j]=1 当且仅当本人被选中（x=1）且序号比其小者恰有 j 人
    # （smaller==j）：用 rank 布尔量通道 smaller==j，再与 x 合取。
    order = [
        [m.NewIntVar(0, max(R - 1, 0), f"order_{p}_{j}") for p in range(P)]
        for j in range(required)
    ]
    pos: List[List[List[cp_model.IntVar]]] = []
    for r in range(R):
        per_paper: List[List[cp_model.IntVar]] = []
        for p in range(P):
            smaller = sum(x[s][p] for s in range(r))
            per_j: List[cp_model.IntVar] = []
            for j in range(required):
                rank = m.NewBoolVar(f"rank_{r}_{p}_{j}")
                m.Add(smaller == j).OnlyEnforceIf(rank)
                m.Add(smaller != j).OnlyEnforceIf(rank.Not())
                z = m.NewBoolVar(f"pos_{r}_{p}_{j}")
                m.AddBoolAnd([x[r][p], rank]).OnlyEnforceIf(z)
                m.AddBoolOr([x[r][p].Not(), rank.Not()]).OnlyEnforceIf(z.Not())
                m.Add(order[j][p] == r).OnlyEnforceIf(z)
                m.Add(order[j][p] != r).OnlyEnforceIf(z.Not())
                per_j.append(z)
            per_paper.append(per_j)
        pos.append(per_paper)

    # 每个 (p, j) 恰有一名审阅人占位：选中集合中第 j 小序号唯一确定
    # （未选中者 x=0，其 pos 恒为 0）；AllDifferent 防御性保证位置互异
    for p in range(P):
        for j in range(required):
            m.AddExactlyOne(pos[r][p][j] for r in range(R))
        m.AddAllDifferent(order[j][p] for j in range(required))

    return m, {
        "x": x,
        "loads": loads,
        "max_load": max_load,
        "devs": devs,
        "dev_sum": sum(devs),
        "order": order,
        "D": D,
    }


def _solver(seed: int) -> cp_model.CpSolver:
    s = cp_model.CpSolver()
    s.parameters.num_search_workers = 1  # 单线程保证确定性
    s.parameters.random_seed = seed
    return s


def _assignment(
    solver: cp_model.CpSolver,
    ids: List[Any],
    order: List[List[cp_model.IntVar]],
    required: int,
) -> List[List[Any]]:
    """从 order 变量读取逐卷审阅人编号（每卷升序）。"""
    P = len(order[0])
    papers: List[List[Any]] = []
    for p in range(P):
        rids = [ids[int(solver.Value(order[j][p]))] for j in range(required)]
        papers.append(sorted(rids, key=_id_sort_key))
    return papers


def _load_list(
    solver: cp_model.CpSolver, ids: List[Any], loads: List[cp_model.LinearExpr]
) -> List[Dict[str, Any]]:
    return [
        {"reviewer_id": rid, "load": int(solver.Value(loads[r]))}
        for r, rid in enumerate(ids)
    ]


# --------------------------------------------------------------------------- #
# 无解诊断（只做人力充分性统计；最终可行性以 CP-SAT 判定为准）
# --------------------------------------------------------------------------- #
def _infeasible_result(
    ids: List[Any],
    capacities: List[int],
    forbidden: List[set],
    P: int,
    required: int,
) -> Dict[str, Any]:
    papers: List[Dict[str, Any]] = []
    for p in range(P):
        eligible = [ids[r] for r in range(len(ids)) if p not in forbidden[r]]
        short = max(0, required - len(eligible))
        papers.append(
            {
                "paper": p,
                "required": required,
                "available_count": len(eligible),
                "available_reviewer_ids": eligible,
                "shortage": short,
            }
        )
    total_demand = P * required
    total_capacity = sum(capacities)
    return {
        "status": "infeasible",
        "message": "不存在满足回避关系与总容量约束的盲审分配方案",
        "papers": papers,
        "total_demand": total_demand,
        "total_capacity": total_capacity,
        "total_capacity_shortage": max(0, total_demand - total_capacity),
    }


# --------------------------------------------------------------------------- #
# 入口：分层字典序择优
# --------------------------------------------------------------------------- #
def solve_review_plan(
    paper_count: int,
    reviewers: List[Dict[str, Any]],
    required: int,
    seed: int = 42,
) -> Dict[str, Any]:
    """盲审联合分配。

    paper_count 为源任务卷数（1~5），reviewers 为 [{id, capacity,
    forbidden_papers}]，required 为每卷所需审阅人数。可行时返回完整方案
    （逐卷审阅人编号、逐人负载与择优指标）；无解时返回 infeasible 及
    各卷可用人力与总容量缺口，不返回任何部分方案。
    """
    P = int(paper_count)
    ids, capacities, forbidden = _read_x(reviewers, P)
    R = len(ids)
    required = int(required)

    # 无审阅人（schema 层 min_length=1 已拦截）：直接给无解诊断，避免除零
    if R == 0:
        return _infeasible_result(ids, capacities, forbidden, P, required)

    def build() -> Tuple[cp_model.CpModel, Dict[str, Any]]:
        return _build_model(R, P, required, capacities, forbidden)

    # 阶段 1：最大个人负载最小
    m, extra = build()
    m.Minimize(extra["max_load"])
    s = _solver(seed)
    s.parameters.max_time_in_seconds = STAGE_TIME_LIMIT
    status = s.Solve(m)
    if status == cp_model.INFEASIBLE:
        return _infeasible_result(ids, capacities, forbidden, P, required)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": "error", "error": "求解器在规定时间内无法判定可行性"}
    best_max = int(s.Value(extra["max_load"]))

    # 阶段 2：各人负载相对平均值的绝对偏差总和（放大 R 倍）最小
    m, extra = build()
    m.Add(extra["max_load"] == best_max)
    m.Minimize(extra["dev_sum"])
    s = _solver(seed)
    s.parameters.max_time_in_seconds = STAGE_TIME_LIMIT
    status = s.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # 退化：仅固定阶段 1 结果重求一个可行解
        m, extra = build()
        m.Add(extra["max_load"] == best_max)
        s = _solver(seed)
        s.parameters.max_time_in_seconds = STAGE_TIME_LIMIT
        if s.Solve(m) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return {"status": "error", "error": "求解器在规定时间内无法判定可行性"}
    best_dev = int(s.Value(extra["dev_sum"]))

    # 阶段 3：按卷号展平的审阅人编号序列最小（R 进制分块，逐块最小化）。
    # 每卷内顺序为第 j 小审阅人序号；块严格落在单卷内部，按 (卷序, 块序)
    # 逐块固定后最小化下一块，唯一破缺排列对称。
    k = _chunk_bits(R)
    n_chunks = (required + k - 1) // k
    fixed: List[Tuple[int, int, int]] = []  # (卷 p, 块 c, 块值)

    def chunk_expr(model_extra: Dict[str, Any], p: int, c: int):
        lo, hi = c * k, min((c + 1) * k, required)
        return sum(
            model_extra["order"][lo + j][p] * (R ** j) for j in range(hi - lo)
        )

    for step in range(P * n_chunks):
        p, c = divmod(step, n_chunks)
        m, extra = build()
        m.Add(extra["max_load"] == best_max)
        m.Add(extra["dev_sum"] == best_dev)
        for fp, fc, fv in fixed:
            m.Add(chunk_expr(extra, fp, fc) == fv)
        e = chunk_expr(extra, p, c)
        m.Minimize(e)
        s = _solver(seed)
        s.parameters.max_time_in_seconds = STAGE_TIME_LIMIT
        status = s.Solve(m)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break  # 保留此前各块已固定的解（上一可行模型）
        fixed.append((p, c, int(s.Value(e))))

    # 用最后一个可行模型的解输出（fixed 全部约束即唯一序列）
    m, extra = build()
    m.Add(extra["max_load"] == best_max)
    m.Add(extra["dev_sum"] == best_dev)
    for fp, fc, fv in fixed:
        m.Add(chunk_expr(extra, fp, fc) == fv)
    s = _solver(seed)
    s.parameters.max_time_in_seconds = STAGE_TIME_LIMIT
    status = s.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # 理论上不会发生（前面阶段已证明可行）；保守报错而非返回部分方案
        return {"status": "error", "error": "求解器无法复现已求得的最优解"}

    D = extra["D"]
    return {
        "status": "feasible",
        "required_per_paper": required,
        "papers": [
            {"paper": p, "reviewer_ids": ids_p}
            for p, ids_p in enumerate(
                _assignment(s, ids, extra["order"], required)
            )
        ],
        "workloads": _load_list(s, ids, extra["loads"]),
        "max_load": best_max,
        "load_deviation_sum_scaled": best_dev,
        "load_deviation_sum": round(best_dev / R, 6),
        "average_load": round(D / R, 6),
        "total_demand": D,
        "total_capacity": sum(capacities),
        "seed": seed,
    }
