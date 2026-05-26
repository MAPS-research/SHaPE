---
pretty_name: "SHAPE"
license: other
language:
  - en
task_categories:
  - question-answering
  - text-generation
tags:
  - education
  - tutoring
  - pedagogy
  - safety
  - knowledge-graph
  - linear-algebra
  - "arxiv:2604.22134"
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: valid_pairs.jsonl
  - config_name: pairs
    data_files:
      - split: train
        path: valid_pairs.jsonl
  - config_name: questions
    data_files:
      - split: train
        path: questions.jsonl
  - config_name: knowledge_graph_edges
    data_files:
      - split: train
        path: knowledge_graph_edges.jsonl
  - config_name: knowledge_graph_nodes
    data_files:
      - split: train
        path: knowledge_graph_nodes.jsonl
  - config_name: knowledge_graph_matrix
    data_files:
      - split: train
        path: adjacency_matrix_knowledge_graph.csv
---

# SHAPE

SHAPE is a benchmark for studying safe, helpful, and pedagogical behavior in educational LLM tutoring. It focuses on pedagogical jailbreaks, where students try to induce direct answers even when their knowledge state indicates missing prerequisites.

Paper: [SHAPE: Unifying Safety, Helpfulness and Pedagogy for Educational LLMs](https://arxiv.org/abs/2604.22134)

## Dataset Contents

This dataset contains the paper-claim reproduction of the SHAPE benchmark. The default split has 9,087 student-question pairs, matching the paper statistics exactly.

| File | Rows | Description |
|---|---:|---|
| `valid_pairs.jsonl` | 9,087 | Hugging Face-friendly student-question pairs. This is the default config. |
| `valid_pairs.json` | 9,087 | Exact JSON-array output from `python -m pipeline.generate_pairs`. |
| `questions.jsonl` | 1,786 | Linear Algebra question records with required knowledge points. |
| `LinearAlgebra_hard_ds_steps_parallel.json` | 1,786 | Original JSON Lines question file used by the repo. |
| `LinearAlgebra_hard_random100_1.json` | 100 | Original 100-question subset. |
| `knowledge_graph_nodes.jsonl` | 91 | Knowledge graph nodes with direct prerequisites/dependents under the repo-code interpretation. |
| `knowledge_graph_edges.jsonl` | 87 | Knowledge graph edges under the repo-code interpretation. |
| `adjacency_matrix_knowledge_graph.csv` | 91 x 91 | Original adjacency matrix. |
| `dataset_summary.json` | 1 | Machine-readable generation and validation summary. |
| `DATA_VALIDATION.md` | 1 | Human-readable upload review report. |

## Loading

```python
from datasets import load_dataset

pairs = load_dataset("your-hf-org-or-username/SHAPE")
questions = load_dataset("your-hf-org-or-username/SHAPE", "questions")
kg_edges = load_dataset("your-hf-org-or-username/SHAPE", "knowledge_graph_edges")
kg_nodes = load_dataset("your-hf-org-or-username/SHAPE", "knowledge_graph_nodes")
```

The default config loads `valid_pairs.jsonl` into the `train` split.

## Schema

### `valid_pairs.jsonl`

Each row is one generated student-question pair.

- `pair_id`: Stable row id.
- `question_idx`: Index into the 1,786-question source file.
- `problem`: Question text.
- `expected_answer`: Reference answer.
- `required_kps`: Knowledge points required by the question, as used by the repo generation script.
- `missing_kps`: Student's missing knowledge points for this pair.
- `mastered_kps`: Required knowledge points treated as mastered for this pair.
- `num_required_kps`, `num_missing_kps`, `num_mastered_kps`: Derived counts.
- `has_missing_kps`: Whether the pair should trigger tutoring instead of direct answering.
- `unknown_required_kps`: Required KPs not present in the 91-node graph.

### Knowledge Graph Direction

The pair generation follows the repository implementation. In that implementation, `adjacency_matrix[row][column] = 1` means `row_kp` depends on `column_kp`; therefore `column_kp` is interpreted as the prerequisite of `row_kp`.

`knowledge_graph_edges.jsonl` uses:

- `prerequisite_kp`: The prerequisite under the repo-code interpretation.
- `dependent_kp`: The dependent knowledge point under the repo-code interpretation.
- `matrix_row_kp`, `matrix_column_kp`: The original matrix labels.

## Reproducing the Paper Claim

From the project root:

```bash
python -m pipeline.generate_pairs \
  --questions data/LinearAlgebra_hard_ds_steps_parallel.json \
  --adjacency data/adjacency_matrix_knowledge_graph.csv \
  --output data/valid_pairs.json
```

Expected summary:

```text
Total questions: 1786
Total candidate (q, sR) pairs: 12238
Valid (q, sR) pairs: 9087
Student has knowledge gaps (should tutor): 7301
Student mastered all KPs (should answer): 1786
```

## Intended Use

SHAPE is intended for research on educational LLM safety, tutoring behavior, mastery-aware response gating, and robustness against answer-inducing prompts.

## Limitations

This release preserves the repository's paper-claim generation exactly. The following are audit notes, not alternate dataset definitions:

- Some source questions contain repeated entries in `step`; these are preserved because they affect the published 9,087-pair count.
- A small number of required KPs in question records are not present in the 91-node graph; the repo implementation treats them as having no graph prerequisites.

See `DATA_VALIDATION.md` for the full reproduction log and diagnostics.

## Citation

```bibtex
@inproceedings{zhao2026shape,
  title={SHAPE: Unifying Safety, Helpfulness and Pedagogy for Educational LLMs},
  author={Zhao, Sihang and Yu, Kangrui and Yuan, Youliang and He, Pinjia and Wen, Hongyi},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026}
}
```
