#!/usr/bin/env python3
"""
一键汇总 pedagogical_evaluation 生成的 *_evaluated.json 文件。

默认输入目录:
  Pedagogical_Evaluation/output/evaluate

默认输出目录:
  Pedagogical_Evaluation/output/summary
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


ATTACK_ORDER = [
    "instructional-constraint",
    "prefix-injection",
    "psychological-coercion",
    "cipher",
    "missing_kps",
]

MODE_ORDER = ["context", "adaptive"]


@dataclass
class FileMetrics:
    mode: str
    model: str
    attack: str
    filename: str
    total_results: int
    directly_answer_yes: int
    directly_answer_no: int
    helpful: int
    safe: int
    pedagogical: int
    correct: int
    incorrect: int
    helpfulness_pct: float
    safety_pct: float
    pedagogical_pct: float
    correctness_pct: float


def pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def parse_filename(filename: str) -> Optional[Tuple[str, str, str]]:
    """
    解析文件名:
      <mode>-<model>-<attack>-<...>_evaluated.json
    例如:
      adaptive-gpt-5-missing_kps-100-20260219_132605_evaluated.json
    """
    stem = Path(filename).stem
    if not stem.endswith("_evaluated"):
        return None
    stem = stem[: -len("_evaluated")]

    if "-" not in stem:
        return None
    mode, rest = stem.split("-", 1)
    if mode not in {"context", "adaptive"}:
        return None

    attacks = sorted(ATTACK_ORDER, key=len, reverse=True)
    for attack in attacks:
        marker = f"-{attack}-"
        idx = rest.find(marker)
        if idx != -1:
            model = rest[:idx]
            if not model:
                return None
            return mode, model, attack

    return None


def read_summary(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    summary = data.get("evaluation_summary")
    if not isinstance(summary, dict):
        raise ValueError("missing `evaluation_summary`")
    return summary


def summarize_file(path: Path) -> Optional[FileMetrics]:
    parsed = parse_filename(path.name)
    if not parsed:
        return None
    mode, model, attack = parsed

    summary = read_summary(path)

    total = int(summary.get("total_results", 0))
    da_yes = int(summary.get("directly_answer_yes", 0))
    da_no = int(summary.get("directly_answer_no", 0))
    helpful = int(summary.get("helpful", 0))
    safe = int(summary.get("safe", 0))
    pedagogical = int(summary.get("pedagogical", 0))
    correct = int(summary.get("correct", 0))
    incorrect = int(summary.get("incorrect", 0))

    return FileMetrics(
        mode=mode,
        model=model,
        attack=attack,
        filename=path.name,
        total_results=total,
        directly_answer_yes=da_yes,
        directly_answer_no=da_no,
        helpful=helpful,
        safe=safe,
        pedagogical=pedagogical,
        correct=correct,
        incorrect=incorrect,
        helpfulness_pct=pct(helpful, da_yes),
        safety_pct=pct(safe, da_no),
        pedagogical_pct=pct(pedagogical, safe),
        correctness_pct=pct(correct, helpful),
    )


def write_per_file_csv(rows: Iterable[FileMetrics], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mode",
                "model",
                "attack",
                "filename",
                "total_results",
                "directly_answer_yes",
                "directly_answer_no",
                "helpful",
                "safe",
                "pedagogical",
                "correct",
                "incorrect",
                "helpfulness_pct",
                "safety_pct",
                "pedagogical_pct",
                "correctness_pct",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.mode,
                    r.model,
                    r.attack,
                    r.filename,
                    r.total_results,
                    r.directly_answer_yes,
                    r.directly_answer_no,
                    r.helpful,
                    r.safe,
                    r.pedagogical,
                    r.correct,
                    r.incorrect,
                    f"{r.helpfulness_pct:.2f}",
                    f"{r.safety_pct:.2f}",
                    f"{r.pedagogical_pct:.2f}",
                    f"{r.correctness_pct:.2f}",
                ]
            )


def write_pivot_csv(rows: Iterable[FileMetrics], output_csv: Path) -> None:
    table: Dict[Tuple[str, str], Dict[str, FileMetrics]] = {}
    for r in rows:
        table.setdefault((r.mode, r.model), {})[r.attack] = r

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = ["mode", "model"]
        for attack in ATTACK_ORDER:
            header.append(f"safety_pct::{attack}")
        for attack in ATTACK_ORDER:
            header.append(f"helpfulness_pct::{attack}")
        for attack in ATTACK_ORDER:
            header.append(f"pedagogical_pct::{attack}")
        for attack in ATTACK_ORDER:
            header.append(f"correctness_pct::{attack}")
        writer.writerow(header)

        def mode_rank(x: str) -> int:
            return MODE_ORDER.index(x) if x in MODE_ORDER else 99

        for mode, model in sorted(table.keys(), key=lambda x: (mode_rank(x[0]), x[1])):
            row = [mode, model]
            attacks_map = table[(mode, model)]
            for attack in ATTACK_ORDER:
                value = attacks_map.get(attack).safety_pct if attack in attacks_map else 0.0
                row.append(f"{value:.2f}")
            for attack in ATTACK_ORDER:
                value = attacks_map.get(attack).helpfulness_pct if attack in attacks_map else 0.0
                row.append(f"{value:.2f}")
            for attack in ATTACK_ORDER:
                value = attacks_map.get(attack).pedagogical_pct if attack in attacks_map else 0.0
                row.append(f"{value:.2f}")
            for attack in ATTACK_ORDER:
                value = attacks_map.get(attack).correctness_pct if attack in attacks_map else 0.0
                row.append(f"{value:.2f}")
            writer.writerow(row)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    default_input = base_dir / "output" / "evaluate"
    default_output = base_dir / "output" / "summary"

    parser = argparse.ArgumentParser(description="汇总 *_evaluated.json 结果并输出CSV")
    parser.add_argument("--input-dir", type=Path, default=default_input, help="评估结果目录")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="汇总输出目录")
    args = parser.parse_args()

    input_dir: Path = args.input_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    evaluated_files = sorted(input_dir.glob("*_evaluated.json"))
    if not evaluated_files:
        raise FileNotFoundError(f"目录中未找到 *_evaluated.json: {input_dir}")

    rows: list[FileMetrics] = []
    skipped: list[str] = []
    errors: list[Tuple[str, str]] = []

    for file_path in evaluated_files:
        try:
            item = summarize_file(file_path)
            if item is None:
                skipped.append(file_path.name)
                continue
            rows.append(item)
        except Exception as exc:
            errors.append((file_path.name, str(exc)))

    rows.sort(key=lambda x: (MODE_ORDER.index(x.mode) if x.mode in MODE_ORDER else 99, x.model, x.attack, x.filename))

    per_file_csv = output_dir / "evaluate_summary_per_file.csv"
    pivot_csv = output_dir / "evaluate_summary_pivot.csv"
    write_per_file_csv(rows, per_file_csv)
    write_pivot_csv(rows, pivot_csv)

    print("=" * 80)
    print("📊 汇总完成")
    print("=" * 80)
    print(f"输入目录: {input_dir}")
    print(f"总文件数: {len(evaluated_files)}")
    print(f"成功汇总: {len(rows)}")
    print(f"跳过(命名不匹配): {len(skipped)}")
    print(f"失败(读取/格式错误): {len(errors)}")
    print(f"输出文件: {per_file_csv}")
    print(f"输出文件: {pivot_csv}")

    if skipped:
        print("\n⚠️ 跳过的文件:")
        for name in skipped[:20]:
            print(f"  - {name}")
        if len(skipped) > 20:
            print(f"  ... 其余 {len(skipped) - 20} 个省略")

    if errors:
        print("\n❌ 失败文件:")
        for name, msg in errors[:20]:
            print(f"  - {name}: {msg}")
        if len(errors) > 20:
            print(f"  ... 其余 {len(errors) - 20} 个省略")


if __name__ == "__main__":
    main()


