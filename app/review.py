"""人工复核出没组挑选（纯逻辑，无 I/O）。

从某个已发布版本的训练索引候选组中，选出满足各物种配额的最少组集合：

* 候选组：指定相机、组起始 UTC 日期（组内最早照片的拍摄日期）等于指定日期的组。
* 配额：每个物种至少被 N 个不同的组覆盖；同组含多种物种时可同时抵多个
  物种的配额，同组同物种多张照片仍只算一组。
* 目标：满足全部配额且组数最少；并列时按 group_id 字典序取最小组合
  （候选按 group_id 升序排列后按组合大小升序枚举，第一个可行解即稳定答案，
  与求解时刻、并发无关）。
* 保护：候选组数超过 MAX_CANDIDATES(18) 时不求解，提示缩小范围。
* 无解时区分原因：候选中某物种覆盖组数已小于配额 -> 可用组不足；
  覆盖足够但最小可行组数超过总组数上限 -> 总上限不足。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

MAX_CANDIDATES = 18


@dataclass(frozen=True)
class CandidateGroup:
    group_id: str
    started_at: str  # 组内最早照片的 captured_at（ISO8601 UTC）
    species: frozenset[str]
    photo_count: int


class TooManyCandidates(Exception):
    """候选组数超过上限，需缩小范围后再试。"""

    def __init__(self, count: int, limit: int = MAX_CANDIDATES):
        self.count = count
        self.limit = limit
        super().__init__(f"当日候选 {count} 组，超过 {limit} 组上限")


class Infeasible(Exception):
    """无解。reason: "shortage"（可用组不足）或 "cap"（总上限不足）。"""

    def __init__(self, reason: str, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(detail)


def select_groups(
    candidates: list[CandidateGroup],
    quotas: dict[str, int],
    max_groups: int,
    max_candidates: int = MAX_CANDIDATES,
) -> list[str]:
    """返回满足配额的最少 group_id 列表（按 group_id 升序）。

    候选超过 max_candidates 抛 TooManyCandidates；无解抛 Infeasible
    （detail 中说明是可用组不足还是总上限不足）；quotas 全为 0 抛 ValueError。
    """
    quotas = {s: n for s, n in quotas.items() if n > 0}
    if not quotas:
        raise ValueError("至少需要一个物种配额大于 0")

    ordered = sorted(candidates, key=lambda g: g.group_id)
    if len(ordered) > max_candidates:
        raise TooManyCandidates(len(ordered), max_candidates)

    species_list = sorted(quotas)
    need = tuple(quotas[s] for s in species_list)
    sidx = {s: i for i, s in enumerate(species_list)}
    hits = [tuple(sidx[s] for s in g.species if s in sidx) for g in ordered]
    n = len(species_list)

    # 可用组不足：某物种在候选中的覆盖组数已小于配额，即使全选也无解
    cover = [0] * n
    for h in hits:
        for si in h:
            cover[si] += 1
    short = [(species_list[i], need[i], cover[i]) for i in range(n) if cover[i] < need[i]]
    if short:
        detail = "；".join(f"物种 {s} 需 {n_} 组、候选中仅 {c} 组" for s, n_, c in short)
        raise Infeasible("shortage", f"可用组不足：{detail}")

    def feasible(combo: tuple[int, ...]) -> bool:
        counts = [0] * n
        for gi in combo:
            for si in hits[gi]:
                counts[si] += 1
        return all(counts[i] >= need[i] for i in range(n))

    # 按组数升序枚举；combinations 按字典序产生组合，候选已按 group_id 排序，
    # 故第一个可行解就是"组数最少、并列按组号字典序最小"的稳定答案。
    limit = min(max_groups, len(ordered))
    for size in range(1, limit + 1):
        for combo in itertools.combinations(range(len(ordered)), size):
            if feasible(combo):
                return [ordered[gi].group_id for gi in combo]

    # 覆盖检查已保证全集可行，能到这里说明限制在 max_groups 内无解 -> 总上限不足。
    # 求出最小可行组数，帮助研究员调整上限。
    for size in range(limit + 1, len(ordered) + 1):
        for combo in itertools.combinations(range(len(ordered)), size):
            if feasible(combo):
                raise Infeasible(
                    "cap",
                    f"总上限不足：满足配额至少需 {size} 组，超过总组数上限 {max_groups}",
                )
    raise AssertionError("unreachable: 覆盖检查保证全集可行")  # pragma: no cover
