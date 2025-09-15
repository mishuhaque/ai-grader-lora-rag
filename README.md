# ai-grader-lora-rag
# AI Grader with LoRA + RAG

🚀 An AI-powered grading assistant that combines **LoRA fine-tuning** and **RAG (Retrieval-Augmented Generation)** to evaluate student answers using course rubrics and lectures.

## Features
- **LoRA fine-tuning** on grading data to capture instructor scoring style
- **RAG pipeline** to retrieve up-to-date lecture notes and rubrics
- Prevents hallucinations by grounding answers in faculty material
- Lightweight deployment with FastAPI

## Project Flow
1. Collect lectures, rubrics, and sample graded answers.
2. Tokenize and prepare data.
3. Train LoRA adapters on grading dataset (`src/lora_trainer.py`).
4. Build RAG pipeline (`src/rag_pipeline.py`) to retrieve rubric/lecture context.
5. Run demo app (`src/app.py`) to grade student answers.

## Quickstart
```bash
git clone https://github.com/mishuhaque/ai-grader-lora-rag.git
cd ai-grader-lora-rag
pip install -r requirements.txt
