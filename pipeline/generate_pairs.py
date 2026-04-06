#!/usr/bin/env python3
"""
Generate valid (question, student-state) pairs for the SHAPE benchmark.

Implements Algorithm 1 from the paper:
  1. For each question q, extract the required knowledge points R_q.
  2. Enumerate all 2^|R_q| subsets of R_q as candidate missing-KP sets.
  3. Filter out invalid combinations using prerequisite consistency (Eq. 18):
     a missing KP cannot be a prerequisite of a mastered KP.
  4. Output the valid pairs.

Usage:
    python -m pipeline.generate_pairs \
        --questions data/LinearAlgebra_hard_ds_steps_parallel.json \
        --adjacency data/adjacency_matrix_knowledge_graph.csv \
        --output data/valid_pairs.json

    # Statistics only (no output file):
    python -m pipeline.generate_pairs \
        --questions data/LinearAlgebra_hard_ds_steps_parallel.json \
        --adjacency data/adjacency_matrix_knowledge_graph.csv \
        --stats-only
"""

import argparse
import itertools
import json
import os
import sys
from collections import Counter
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from adaptive_tutor.knowledge_graph import KnowledgeGraph


def load_questions(path: str) -> List[Dict[str, Any]]:
    """Load questions from JSON Lines file."""
    questions = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def generate_valid_pairs(
    questions: List[Dict[str, Any]],
    knowledge_graph: KnowledgeGraph,
) -> List[Dict[str, Any]]:
    """
    Generate all valid (question, student-state) pairs.

    For each question, enumerates all subsets of required KPs as candidate
    missing-KP sets, then filters by prerequisite consistency.

    Args:
        questions: List of question dicts (must have 'step' field).
        knowledge_graph: KnowledgeGraph instance for validation.

    Returns:
        List of dicts, each with:
          - question_idx: index into the questions list
          - problem: the question text
          - required_kps: knowledge points required for this question
          - missing_kps: the student's missing KPs (valid subset)
          - mastered_kps: the student's mastered KPs (complement)
    """
    pairs = []
    for q_idx, question in enumerate(questions):
        required_kps = question.get('step', [])
        n = len(required_kps)

        for r in range(n + 1):
            for combo in itertools.combinations(required_kps, r):
                missing_kps = list(combo)
                try:
                    if knowledge_graph.is_valid_missing_combination(required_kps, missing_kps):
                        mastered_kps = [kp for kp in required_kps if kp not in missing_kps]
                        pairs.append({
                            'question_idx': q_idx,
                            'problem': question.get('problem', ''),
                            'expected_answer': question.get('answer', ''),
                            'required_kps': required_kps,
                            'missing_kps': missing_kps,
                            'mastered_kps': mastered_kps,
                        })
                except Exception:
                    pass

    return pairs


def print_statistics(questions: List[Dict[str, Any]], pairs: List[Dict[str, Any]]) -> None:
    """Print statistics matching the paper's format."""
    rq_counter = Counter()
    for q in questions:
        rq_counter[len(q.get('step', []))] += 1

    total_candidates = 0
    print(f"Total questions: {len(questions)}")
    print(f"\nCandidate enumeration (before filtering):")
    for rq_size in sorted(rq_counter.keys()):
        count = rq_counter[rq_size]
        candidates = count * (2 ** rq_size)
        total_candidates += candidates
        print(f"  |Rq| = {rq_size}: {count} questions => {count} * 2^{rq_size} = {candidates}")
    print(f"  Total candidate (q, sR) pairs: {total_candidates}")

    print(f"\nAfter prerequisite-consistency filtering (Eq. 18):")
    print(f"  Valid (q, sR) pairs: {len(pairs)}")

    # Breakdown of valid pairs
    has_missing = sum(1 for p in pairs if len(p['missing_kps']) > 0)
    no_missing = sum(1 for p in pairs if len(p['missing_kps']) == 0)
    print(f"    - Student has knowledge gaps (should tutor): {has_missing}")
    print(f"    - Student mastered all KPs (should answer):  {no_missing}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate valid (question, student-state) pairs for the SHAPE benchmark.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--questions', type=str, required=True,
                        help='Path to questions JSON Lines file')
    parser.add_argument('--adjacency', type=str, required=True,
                        help='Path to knowledge graph adjacency matrix CSV')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path (omit for stats only)')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only print statistics, do not write output')

    args = parser.parse_args()

    # Load data
    kg = KnowledgeGraph(args.adjacency)
    questions = load_questions(args.questions)

    # Generate pairs
    pairs = generate_valid_pairs(questions, kg)

    # Print statistics
    print_statistics(questions, pairs)

    # Write output
    if not args.stats_only and args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(pairs)} pairs to: {args.output}")
    elif not args.stats_only and not args.output:
        print("\nNo --output specified. Use --output to save pairs, or --stats-only for statistics.")


if __name__ == '__main__':
    main()
