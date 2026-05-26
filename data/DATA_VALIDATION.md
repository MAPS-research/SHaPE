# SHAPE Data Validation Report

Generated on 2026-05-26 from the repository data and `pipeline.generate_pairs`.

## Paper-Claim Reproduction

The repository command reproduces the paper/README claim:

```bash
python -m pipeline.generate_pairs \
  --questions data/LinearAlgebra_hard_ds_steps_parallel.json \
  --adjacency data/adjacency_matrix_knowledge_graph.csv \
  --stats-only
```

Observed output:

```text
Total questions: 1786

Candidate enumeration (before filtering):
  |Rq| = 1: 281 questions => 281 * 2^1 = 562
  |Rq| = 2: 135 questions => 135 * 2^2 = 540
  |Rq| = 3: 1348 questions => 1348 * 2^3 = 10784
  |Rq| = 4: 22 questions => 22 * 2^4 = 352
  Total candidate (q, sR) pairs: 12238

After prerequisite-consistency filtering (Eq. 18):
  Valid (q, sR) pairs: 9087
    - Student has knowledge gaps (should tutor): 7301
    - Student mastered all KPs (should answer):  1786
```

Generated file:

- `valid_pairs.json`: 9,087 rows, exact JSON-array output from the repo script.
- `valid_pairs.jsonl`: 9,087 rows, Hugging Face-friendly row format derived from `valid_pairs.json`.

## Upload-Readiness Checks

| Check | Result |
|---|---:|
| Source questions | 1,786 |
| Knowledge graph nodes | 91 |
| Knowledge graph edges, repo-code direction | 87 |
| Candidate pairs before filtering | 12,238 |
| Valid student-question pairs | 9,087 |
| Pairs with knowledge gaps | 7,301 |
| Pairs with all required KPs mastered | 1,786 |
| Questions represented in generated pairs | 1,786 |
| Minimum generated pairs per question | 2 |
| Maximum generated pairs per question | 16 |
| `missing_kps` subset violations | 0 |

## Distribution

Candidate enumeration by required-KP count:

| `|Rq|` | Questions | Candidate pairs |
|---:|---:|---:|
| 1 | 281 | 562 |
| 2 | 135 | 540 |
| 3 | 1,348 | 10,784 |
| 4 | 22 | 352 |

Valid pairs by required-KP count:

| Required KPs | Valid pairs |
|---:|---:|
| 1 | 562 |
| 2 | 488 |
| 3 | 7,788 |
| 4 | 249 |

Valid pairs by missing-KP count:

| Missing KPs | Valid pairs |
|---:|---:|
| 0 | 1,786 |
| 1 | 2,812 |
| 2 | 3,051 |
| 3 | 1,416 |
| 4 | 22 |

## Additional Diagnostics, Not Release-Defining

The release target is the repository/paper reproduction above: 9,087 valid student state-question pairs. The following diagnostics explain why other counts can appear during audit; they are not used to define the uploaded dataset.

1. The released question file contains repeated `step` entries in 85 questions, with 86 extra repeated occurrences. The repo generation script preserves these repeated entries, and preserving them is required to reproduce the paper's 9,087-pair claim. A separate cleaning pass that deduplicates each question's required KPs would produce 8,811 unique logical pairs, but that is not the paper-claim release.

2. There are 22 questions with required KPs that are not in the 91-node knowledge graph. The repo implementation treats unknown KPs as having no graph prerequisites, and this behavior is part of the 9,087-pair reproduction. The missing graph labels are:

| Required KP not in graph | Occurrences |
|---|---:|
| The Matrix of a Linear Transformation Relative to a Basis | 8 |
| Sets and Functions | 6 |
| Triangular Matrices | 3 |
| Statements and Predicates | 2 |
| Classifying Quadratic Curves | 1 |
| Jordan Canonical Form of a 2x2 Matrix | 1 |
| Linear Independence in Abstract Vector Spaces | 1 |

3. The adjacency matrix direction used for the 9,087-pair claim is the repository-code interpretation: `adjacency_matrix[row][column] = 1` means `row_kp` depends on `column_kp`. Reversing the matrix direction is an alternate interpretation and would produce 9,103 valid pairs, which does not match the paper claim.

## Recommendation Before Hugging Face Upload

Upload the current files to publish the benchmark exactly as claimed in the paper and README. If a cleaned follow-up release is desired later, create it as a separate version/config after deciding whether to deduplicate per-question `step` lists and how to handle the seven required KP labels absent from the graph.
