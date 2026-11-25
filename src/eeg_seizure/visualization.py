from __future__ import annotations

from collections.abc import Iterable


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping or adjacent time intervals."""
    ordered = sorted(intervals)
    if not ordered:
        return []

    merged: list[tuple[float, float]] = []
    start, end = ordered[0]
    if end < start:
        raise ValueError("Interval end must not precede its start")

    for next_start, next_end in ordered[1:]:
        if next_end < next_start:
            raise ValueError("Interval end must not precede its start")
        if next_start <= end:
            end = max(end, next_end)
        else:
            merged.append((start, end))
            start, end = next_start, next_end
    merged.append((start, end))
    return merged
