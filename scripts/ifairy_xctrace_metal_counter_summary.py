#!/usr/bin/env python3

from __future__ import annotations

import argparse
import bisect
import gzip
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


INSTANCE_SUFFIX = re.compile(r" \(\d+\)$")

DEFAULT_COUNTERS = (
    "Kernel Occupancy",
    "Instruction Throughput Limiter",
    "Instruction Throughput Utilization",
    "Compute Shader Launch Limiter",
    "ALU Utilization",
    "F32 Limiter",
    "F32 Utilization",
    "F16 Limiter",
    "F16 Utilization",
    "Integer and Conditional Limiter",
    "Integer and Conditional Utilization",
    "Control Flow Limiter",
    "Control Flow Utilization",
    "Integer and Complex Limiter",
    "Integer and Complex Utilization",
    "L1 Cache Limiter",
    "L1 Cache Utilization",
    "Buffer L1 Miss Rate",
    "Compute SIMD Groups Inflight",
    "L1 Read Bandwidth",
    "Buffer L1 Read Bandwidth",
    "GPU Bandwidth",
    "GPU Read Bandwidth",
    "GPU Write Bandwidth",
    "MMU Limiter",
    "MMU Utilization",
    "Last Level Cache Limiter",
    "Last Level Cache Utilization",
    "Last Level Cache Bandwidth",
)

ROW_PATTERNS = {
    tag: re.compile(fr"<{tag}\b([^>]*?)(?:>([^<]*)</{tag}>|/>)".encode())
    for tag in ("event-time", "uint32", "fixed-decimal")
}
ID_ATTRIBUTE = re.compile(rb'\bid="(\d+)"')
REF_ATTRIBUTE = re.compile(rb'\bref="(\d+)"')


@dataclass
class Stats:
    count: int = 0
    total: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)


@dataclass(frozen=True)
class CounterInfo:
    name: str
    kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Apple GPU counters sampled while selected Metal shaders run. "
            "Inputs are xctrace XML exports for metal-shader-profiler-intervals, "
            "gpu-counter-info, and gpu-counter-value."
        )
    )
    parser.add_argument("--shader-xml", type=Path, required=True)
    parser.add_argument("--counter-info-xml", type=Path, required=True)
    parser.add_argument(
        "--counter-values-xml",
        type=Path,
        required=True,
        help="gpu-counter-value XML path; .gz files are read transparently",
    )
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        metavar="NAME=REGEX",
        help="shader group and regular expression; may be repeated",
    )
    parser.add_argument(
        "--counters",
        nargs="+",
        default=list(DEFAULT_COUNTERS),
        help="counter names to print (default: high-signal limiter set)",
    )
    parser.add_argument(
        "--all-counters",
        action="store_true",
        help="print every sampled counter instead of --counters",
    )
    return parser.parse_args()


def element_id(elem: ET.Element) -> int | None:
    value = elem.get("id")
    return int(value) if value is not None else None


def resolve_int(elem: ET.Element, values: dict[int, int]) -> int | None:
    ref = elem.get("ref")
    if ref is not None:
        return values.get(int(ref))
    text = (elem.text or "").strip()
    return int(text) if text else None


def resolve_text(elem: ET.Element, values: dict[int, str]) -> str | None:
    ref = elem.get("ref")
    if ref is not None:
        return values.get(int(ref))
    value = (elem.get("fmt") or elem.text or "").strip()
    return value or None


def direct_child(row: ET.Element, tag: str, occurrence: int = 0) -> ET.Element | None:
    found = 0
    for child in row:
        if child.tag != tag:
            continue
        if found == occurrence:
            return child
        found += 1
    return None


def parse_groups(specs: list[str]) -> dict[str, re.Pattern[str]]:
    groups: dict[str, re.Pattern[str]] = {}
    for spec in specs:
        name, separator, pattern = spec.partition("=")
        if not separator or not name or not pattern:
            raise ValueError(f"invalid --group {spec!r}; expected NAME=REGEX")
        if name in groups:
            raise ValueError(f"duplicate group name: {name}")
        groups[name] = re.compile(pattern)
    return groups


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def parse_shader_intervals(
    path: Path, groups: dict[str, re.Pattern[str]]
) -> dict[str, list[tuple[int, int]]]:
    starts: dict[int, int] = {}
    durations: dict[int, int] = {}
    labels: dict[int, str] = {}
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag == "start-time":
            value = resolve_int(elem, starts)
            elem_key = element_id(elem)
            if elem_key is not None and value is not None:
                starts[elem_key] = value
        elif elem.tag == "duration":
            value = resolve_int(elem, durations)
            elem_key = element_id(elem)
            if elem_key is not None and value is not None:
                durations[elem_key] = value
        elif elem.tag == "metal-object-label":
            value = resolve_text(elem, labels)
            elem_key = element_id(elem)
            if elem_key is not None and value is not None:
                labels[elem_key] = value
        elif elem.tag == "row":
            start_elem = direct_child(elem, "start-time")
            duration_elem = direct_child(elem, "duration")
            label_elem = direct_child(elem, "metal-object-label")
            if start_elem is not None and duration_elem is not None and label_elem is not None:
                start = resolve_int(start_elem, starts)
                duration = resolve_int(duration_elem, durations)
                label = resolve_text(label_elem, labels)
                if start is not None and duration is not None and label is not None:
                    shader = INSTANCE_SUFFIX.sub("", label)
                    for group, pattern in groups.items():
                        if pattern.search(shader):
                            intervals[group].append((start, start + duration))
            elem.clear()

    return {name: merge_intervals(intervals[name]) for name in groups}


def parse_counter_info(path: Path) -> dict[int, CounterInfo]:
    integers: dict[int, int] = {}
    names: dict[int, str] = {}
    strings: dict[int, str] = {}
    counters: dict[int, CounterInfo] = {}

    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag == "uint32":
            value = resolve_int(elem, integers)
            elem_key = element_id(elem)
            if elem_key is not None and value is not None:
                integers[elem_key] = value
        elif elem.tag == "gpu-counter-name":
            value = resolve_text(elem, names)
            elem_key = element_id(elem)
            if elem_key is not None and value is not None:
                names[elem_key] = value
        elif elem.tag == "string":
            value = resolve_text(elem, strings)
            elem_key = element_id(elem)
            if elem_key is not None and value is not None:
                strings[elem_key] = value
        elif elem.tag == "row":
            counter_id_elem = direct_child(elem, "uint32")
            name_elem = direct_child(elem, "gpu-counter-name")
            kind_elem = direct_child(elem, "string", 1)
            if counter_id_elem is not None and name_elem is not None and kind_elem is not None:
                counter_id = resolve_int(counter_id_elem, integers)
                name = resolve_text(name_elem, names)
                kind = resolve_text(kind_elem, strings)
                if counter_id is not None and name is not None and kind is not None:
                    counters[counter_id] = CounterInfo(name=name, kind=kind)
            elem.clear()

    return counters


def timestamp_in_intervals(
    timestamp: int, intervals: list[tuple[int, int]], starts: list[int]
) -> bool:
    index = bisect.bisect_right(starts, timestamp) - 1
    return index >= 0 and timestamp < intervals[index][1]


def open_xml(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def parse_row_element(
    line: bytes, tag: str
) -> tuple[int | None, int | None, bytes | None] | None:
    match = ROW_PATTERNS[tag].search(line)
    if match is None:
        return None

    attributes = match.group(1)
    elem_id_match = ID_ATTRIBUTE.search(attributes)
    ref_match = REF_ATTRIBUTE.search(attributes)
    elem_id = int(elem_id_match.group(1)) if elem_id_match is not None else None
    ref = int(ref_match.group(1)) if ref_match is not None else None
    return elem_id, ref, match.group(2)


def resolve_row_int(
    parsed: tuple[int | None, int | None, bytes | None] | None,
    values: dict[int, int],
) -> int | None:
    if parsed is None:
        return None
    elem_id, ref, text = parsed
    if ref is not None:
        return values.get(ref)
    if text is None:
        return None
    value = int(text)
    if elem_id is not None:
        values[elem_id] = value
    return value


def cache_counter_uint32_values(
    line: bytes, values: dict[int, int], valid_counter_ids: set[int]
) -> None:
    for match in ROW_PATTERNS["uint32"].finditer(line):
        attributes = match.group(1)
        text = match.group(2)
        if text is None:
            continue
        value = int(text)
        if value not in valid_counter_ids:
            continue
        elem_id_match = ID_ATTRIBUTE.search(attributes)
        if elem_id_match is not None:
            values[int(elem_id_match.group(1))] = value


def matched_groups(
    timestamp: int,
    intervals_by_group: dict[str, list[tuple[int, int]]],
    interval_starts: dict[str, list[int]],
) -> list[str]:
    return [
        group
        for group, intervals in intervals_by_group.items()
        if timestamp_in_intervals(timestamp, intervals, interval_starts[group])
    ]


def collect_relevant_decimal_refs(
    path: Path,
    intervals_by_group: dict[str, list[tuple[int, int]]],
    interval_starts: dict[str, list[int]],
    counter_info: dict[int, CounterInfo],
    selected_counters: set[str] | None,
) -> set[int]:
    timestamps: dict[int, int] = {}
    integers: dict[int, int] = {}
    decimal_refs: set[int] = set()
    valid_counter_ids = set(counter_info)

    with open_xml(path) as source:
        for line in source:
            if b"<row>" not in line:
                continue
            timestamp = resolve_row_int(
                parse_row_element(line, "event-time"), timestamps
            )
            counter_id = resolve_row_int(parse_row_element(line, "uint32"), integers)
            cache_counter_uint32_values(line, integers, valid_counter_ids)
            info = counter_info.get(counter_id) if counter_id is not None else None
            if (
                timestamp is None
                or info is None
                or (selected_counters is not None and info.name not in selected_counters)
                or not matched_groups(timestamp, intervals_by_group, interval_starts)
            ):
                continue

            parsed_value = parse_row_element(line, "fixed-decimal")
            if parsed_value is not None and parsed_value[1] is not None:
                decimal_refs.add(parsed_value[1])

    return decimal_refs


def summarize_counter_values(
    path: Path,
    intervals_by_group: dict[str, list[tuple[int, int]]],
    counter_info: dict[int, CounterInfo],
    selected_counters: set[str] | None,
) -> tuple[dict[str, dict[str, Stats]], dict[str, int]]:
    interval_starts = {
        group: [start for start, _ in intervals]
        for group, intervals in intervals_by_group.items()
    }
    decimal_refs = collect_relevant_decimal_refs(
        path,
        intervals_by_group,
        interval_starts,
        counter_info,
        selected_counters,
    )
    timestamps: dict[int, int] = {}
    integers: dict[int, int] = {}
    decimals: dict[int, float] = {}
    valid_counter_ids = set(counter_info)
    stats: dict[str, dict[str, Stats]] = defaultdict(lambda: defaultdict(Stats))
    sample_timestamps: dict[str, int] = defaultdict(int)
    last_timestamp: dict[str, int | None] = dict.fromkeys(intervals_by_group)

    with open_xml(path) as source:
        for line in source:
            if b"<row>" not in line:
                continue

            timestamp = resolve_row_int(
                parse_row_element(line, "event-time"), timestamps
            )
            counter_id = resolve_row_int(parse_row_element(line, "uint32"), integers)
            cache_counter_uint32_values(line, integers, valid_counter_ids)
            info = counter_info.get(counter_id) if counter_id is not None else None
            groups = (
                matched_groups(timestamp, intervals_by_group, interval_starts)
                if timestamp is not None
                and info is not None
                and (selected_counters is None or info.name in selected_counters)
                else []
            )

            parsed_value = parse_row_element(line, "fixed-decimal")
            if parsed_value is None:
                continue
            elem_id, ref, text = parsed_value
            value = decimals.get(ref) if ref is not None else None
            if ref is None and text is not None and (groups or elem_id in decimal_refs):
                value = float(text)
                if elem_id is not None and elem_id in decimal_refs:
                    decimals[elem_id] = value

            if value is None or info is None:
                continue
            for group in groups:
                stats[group][info.name].add(value)
                if timestamp != last_timestamp[group]:
                    sample_timestamps[group] += 1
                    last_timestamp[group] = timestamp

    return stats, sample_timestamps


def main() -> int:
    args = parse_args()
    try:
        groups = parse_groups(args.group)
    except (ValueError, re.error) as error:
        print(error)
        return 2

    counter_info = parse_counter_info(args.counter_info_xml)
    intervals_by_group = parse_shader_intervals(args.shader_xml, groups)
    missing = [name for name, intervals in intervals_by_group.items() if not intervals]
    if missing:
        print(f"no shader intervals matched group(s): {', '.join(missing)}")
        return 2

    selected = None if args.all_counters else set(args.counters)
    stats, sample_timestamps = summarize_counter_values(
        args.counter_values_xml, intervals_by_group, counter_info, selected
    )

    counter_kind = {info.name: info.kind for info in counter_info.values()}
    for group in groups:
        intervals = intervals_by_group[group]
        shader_ns = sum(end - start for start, end in intervals)
        print(f"[{group}]")
        print(f"shader_intervals: {len(intervals)}")
        print(f"shader_sampled_ms: {shader_ns / 1e6:.3f}")
        print(f"gpu_counter_timestamps: {sample_timestamps[group]}")
        print("mean       min       max  samples  type        counter")
        order = sorted(stats[group]) if args.all_counters else args.counters
        for name in order:
            value = stats[group].get(name)
            if value is None or value.count == 0:
                continue
            print(
                f"{value.total / value.count:8.3f} "
                f"{value.minimum:9.3f} "
                f"{value.maximum:9.3f} "
                f"{value.count:8d}  "
                f"{counter_kind.get(name, ''):10s}  {name}"
            )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
