# LLM Evaluation Framework

A comprehensive Python-based framework for evaluating Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) pipelines.

## Project Structure

```text
llm_evaluation/
├── data/                  # Source documents and evaluation datasets (ignored by Git)
├── evaluation/            # Evaluation logic and metrics
│   ├── judge.py           # LLM-as-a-judge implementation
│   └── metrics.py         # Evaluation metrics (Retrieval, Generation accuracy, etc.)
├── models/                # Schema definitions
│   └── schemas.py         # Pydantic models and schemas
├── outputs/               # Evaluation results and reports (ignored by Git)
├── rag/                   # RAG Pipeline modules
│   ├── generate.py        # Response generation
│   └── ingest.py          # Document ingestion and vector storage
├── .env                   # Environment variables (ignored by Git)
├── .gitignore             # Git ignore configuration
├── main.py                # Main execution script
└── requirements.txt       # Project dependencies
```

## Getting Started

### Prerequisites

- Python 3.9+
- A Google Gemini API Key (or other configured LLM provider keys)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DanishShaikh18/llm_evaluation.git
   cd llm_evaluation
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment:**
   - **Windows (Command Prompt):**
     ```cmd
     venv\Scripts\activate.bat
     ```
   - **Windows (PowerShell):**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Linux/macOS:**
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

## Usage

*Instructions on how to run the ingestion, generation, and evaluation scripts will be added as features are implemented.*
