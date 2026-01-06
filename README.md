# Intelligent Tutoring Systems with Jailbreak Resistance Testing

This repository contains the implementation of two intelligent tutoring approaches and a testing framework for evaluating their robustness against jailbreak attacks.

## Project Structure

```
code/
├── context_based_tutor/     # LangGraph-based intelligent tutor
│   ├── __init__.py
│   ├── run.py              # Main entry point
│   ├── graph.py            # LangGraph workflow definition
│   ├── nodes.py            # Node functions for the workflow
│   ├── state.py            # State type definitions
│   ├── prompts.py          # Prompt templates
│   ├── adapters.py         # LLM adapter utilities
│   └── utils.py            # Utility functions
│
├── adaptive_tutor/          # System-prompt-based intelligent tutor
│   ├── __init__.py
│   ├── run.py              # Main entry point
│   ├── prompts.py          # Adaptive prompt templates
│   └── knowledge_graph.py  # Knowledge graph utilities
│
├── pipeline/                # Testing pipeline
│   ├── __init__.py
│   ├── run_context_test.py # Context tutor testing
│   ├── run_adaptive_test.py # Adaptive tutor testing
│   └── utils.py            # Pipeline utilities
│
├── clients/                 # API clients for various LLMs
│   ├── __init__.py
│   ├── config.py           # Configuration utilities
│   ├── openai_client.py    # OpenAI API client
│   ├── gemini_client.py    # Google Gemini client
│   ├── claude_client.py    # Anthropic Claude client
│   ├── together_ai_client.py # Together AI client
│   ├── openrouter_client.py # OpenRouter client
│   └── educhat_client.py   # EduChat local client
│
├── attack_methods/          # Jailbreak attack prompts
│   ├── refusal_suppression.txt
│   ├── role_play_en.txt
│   ├── role_play_cn.txt
│   └── style_ristrict.txt
│
├── data/                    # Test data
│   ├── adjacency_matrix_knowledge_graph.csv
│   └── LinearAlgebra_hard_random100_1.json
│
├── requirements.txt         # Python dependencies
├── env.example              # Environment variable template
└── README.md               # This file
```

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys:
```bash
cp env.example .env
# Edit .env and add your API keys
```

## Usage

### 1. Context-Based Tutor (Standalone)

The Context-Based Tutor uses a LangGraph workflow to:
1. Decompose questions into steps
2. Map steps to knowledge points
3. Compare required KPs with student's mastered KPs
4. Generate appropriate responses (direct answer or guided tutoring)

```python
from context_based_tutor import run_context_based_tutor

result = run_context_based_tutor(
    question="How do I find the eigenvalues of a matrix?",
    missing_kps=["Eigenvalues and Eigenvectors"],
    adjacency_csv_path="data/adjacency_matrix_knowledge_graph.csv"
)

print(result["final_answer"])
```

### 2. Adaptive Tutor (Standalone)

The Adaptive Tutor embeds knowledge state in the system prompt:

```python
from adaptive_tutor import run_adaptive_tutor
from clients import OpenAIAnswerer

client = OpenAIAnswerer(model="gpt-4o-mini")
result = run_adaptive_tutor(
    question="How do I find eigenvalues?",
    missing_kps=["Eigenvalues and Eigenvectors"],
    client=client,
    adjacency_csv_path="data/adjacency_matrix_knowledge_graph.csv"
)

print(result["answer"])
```

### 3. Running Test Pipeline

#### Context-Based Tutor Test

```bash
python -m pipeline.run_context_test \
    --questions data/LinearAlgebra_hard_random100_1.json \
    --adjacency data/adjacency_matrix_knowledge_graph.csv \
    --model gpt-4o-mini \
    --attack refusal_suppression \
    --start 0 \
    --end 10 \
    --output output
```

#### Adaptive Tutor Test

```bash
python -m pipeline.run_adaptive_test \
    --questions data/LinearAlgebra_hard_random100_1.json \
    --adjacency data/adjacency_matrix_knowledge_graph.csv \
    --model gpt-4o-mini \
    --attack baseline \
    --start 0 \
    --end 10 \
    --output output
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--questions` | Path to questions JSON file | Required |
| `--adjacency` | Path to adjacency matrix CSV | Required |
| `--model` | Model to use (auto-detects provider) | `gpt-4o-mini` |
| `--attack` | Attack method (baseline, refusal_suppression, role_play_en, etc.) | `baseline` |
| `--start` | Start question index | `0` |
| `--end` | End question index | All |
| `--samples` | Samples per question | `2` |
| `--concurrent` | Max concurrent API calls | `10` |
| `--seed` | Random seed | `42` |
| `--output` | Output directory | `output` |

### Attack Methods

| Method | Description |
|--------|-------------|
| `baseline` | No jailbreak attack (standard tutoring) |
| `refusal_suppression` | Prompts that suppress refusal behaviors |
| `role_play_en` | English role-play based attack |

## API Clients

### Supported Providers

| Provider | Models | API Key Variable |
|----------|--------|------------------|
| OpenAI | gpt-5, gpt-5-mini, etc. | `OPENAI_API_KEY` |
| Google | gemini-2.5-pro, gemini-2.5-flash | `GEMINI_API_KEY` |
| Anthropic | claude-4.5-opus, claude-4.5-sonnet, etc. | `CLAUDE_API_KEY` |
| Together AI | Llama, Mistral, Qwen models | `TOGETHER_API_KEY` |
| OpenRouter | Various models | `OPENROUTER_API_KEY` |
| EduChat | Local deployment | `EDUCHAT_BASE_URL` |

### Creating Clients

```python
from clients import (
    OpenAIAnswerer,
    GeminiAnswerer,
    ClaudeAnswerer,
    TogetherAIAnswerer,
    OpenRouterAnswerer,
    EduChatAnswerer
)

# OpenAI
openai_client = OpenAIAnswerer(model="gpt-4o-mini", temperature=0.7)

# Gemini
gemini_client = GeminiAnswerer(model="gemini-1.5-pro")

# Claude
claude_client = ClaudeAnswerer(model="claude-3-sonnet-20240229")

# Together AI
together_client = TogetherAIAnswerer(model="meta-llama/Llama-2-7b-chat-hf")
```

## Data Format

### Questions JSON (JSON Lines format)

Each line contains a JSON object:
```json
{"problem": "Find the eigenvalues of matrix A...", "answer": "λ = 2, 3", "step": ["Eigenvalues and Eigenvectors", "Characteristic Polynomial"]}
```

Fields:
- `problem`: The question text
- `answer`: Expected answer
- `step`: List of knowledge points required

### Adjacency Matrix CSV

Row and column headers are knowledge point names. `1` indicates a prerequisite relationship.

```csv
,KP1,KP2,KP3
KP1,0,1,0
KP2,0,0,1
KP3,0,0,0
```

## Output Format

Test results are saved as JSON with:
- `metadata`: Test configuration and statistics
- `shared_config`: Attack prefix used
- `results`: Array of individual test results

Each result contains:
- `question_id`: Question identifier
- `problem`: Question text
- `expected_answer`: Ground truth
- `model_answer`: Model's response
- `success`: Whether the call succeeded
- `processing_time`: Time taken
- `token_usage`: Estimated token counts
- And more...

## License

This project is released for academic research purposes.

