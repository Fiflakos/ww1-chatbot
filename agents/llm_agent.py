from fastapi import FastAPI
from pydantic import BaseModel

from retrieval_agent import RetrievalAgent
from local_model_generator import LocalHistBERTGenerator

app = FastAPI()

# initialize agents once
retriever = RetrievalAgent(corpus_dir="data_cleaned")
generator = LocalHistBERTGenerator()

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # 1) retrieve top‐3 passages
    hits = retriever.search(req.question, top_k=3)
    # 2) generate via local model
    answer = generator.generate(req.question, hits)
    return AskResponse(answer=answer)
