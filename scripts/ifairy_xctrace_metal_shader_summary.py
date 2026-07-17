#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict


INSTANCE_SUFFIX = re.compile(r" \(\d+\)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Metal Shader Timeline intervals exported by xctrace."
    )
    parser.add_argument("--top", type=int, default=30, help="number of shaders to print")
    parser.add_argument(
        "--prefix",
        default="kernel_",
        help="only include shader names with this prefix (default: kernel_)",
    )
    return parser.parse_args()


def resolve_int(elem: ET.Element, values: dict[int, int]) -> int | None:
    ref = elem.get("ref")
    if ref is not None:
        return values.get(int(ref))
    text = (elem.text or "").strip()
    if not text:
        return None
    return int(text)


def resolve_label(elem: ET.Element, values: dict[int, str]) -> str | None:
    ref = elem.get("ref")
    if ref is not None:
        return values.get(int(ref))
    value = (elem.text or elem.get("fmt") or "").strip()
    return value or None


def main() -> int:
    args = parse_args()
    durations: dict[int, int] = {}
    labels: dict[int, str] = {}
    total_ns: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    min_ns: dict[str, int] = {}
    max_ns: dict[str, int] = defaultdict(int)

    for event, elem in ET.iterparse(sys.stdin.buffer, events=("end",)):
        if elem.tag == "duration":
            elem_id = elem.get("id")
            value = resolve_int(elem, durations)
            if elem_id is not None and value is not None:
                durations[int(elem_id)] = value
        elif elem.tag == "metal-object-label":
            elem_id = elem.get("id")
            value = resolve_label(elem, labels)
            if elem_id is not None and value is not None:
                labels[int(elem_id)] = value
        elif elem.tag == "row":
            duration_elem = elem.find("./duration")
            label_elem = elem.find("./metal-object-label")
            if duration_elem is None or label_elem is None:
                elem.clear()
                continue

            duration_ns = resolve_int(duration_elem, durations)
            label = resolve_label(label_elem, labels)
            if duration_ns is None or label is None:
                elem.clear()
                continue

            name = INSTANCE_SUFFIX.sub("", label)
            if args.prefix and not name.startswith(args.prefix):
                elem.clear()
                continue

            total_ns[name] += duration_ns
            counts[name] += 1
            min_ns[name] = duration_ns if name not in min_ns else min(min_ns[name], duration_ns)
            max_ns[name] = max(max_ns[name], duration_ns)
            elem.clear()

    grand_total = sum(total_ns.values())
    if grand_total == 0:
        print("no matching shader intervals", file=sys.stderr)
        return 2

    print(f"intervals: {sum(counts.values())}")
    print(f"summed_shader_ms: {grand_total / 1e6:.3f}")
    print("share  total_ms  count  avg_us  min_us  max_us  shader")
    for name in sorted(total_ns, key=total_ns.get, reverse=True)[: args.top]:
        total = total_ns[name]
        count = counts[name]
        print(
            f"{100.0 * total / grand_total:5.1f}% "
            f"{total / 1e6:9.3f} "
            f"{count:6d} "
            f"{total / count / 1e3:7.2f} "
            f"{min_ns[name] / 1e3:7.2f} "
            f"{max_ns[name] / 1e3:7.2f}  "
            f"{name}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
