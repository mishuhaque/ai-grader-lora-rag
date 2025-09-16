from fastapi import FastAPI, UploadFile, Form
from rag_pipeline import build_rag

app = FastAPI()

@app.get("/")
def home():
    return {"msg": "AI Grader API is running"}

@app.post("/grade")
async def grade_answer(student_answer: str = Form(...)):
    # For now, just call RAG pipeline
    result = build_rag()
    return {"student_answer": student_answer, "evaluation": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
