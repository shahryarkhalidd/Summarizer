from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = FastAPI()

# Load model & tokenizer
MODEL_PATH = "./urdu-summarizer"
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

# Define request schema
class SummarizeRequest(BaseModel):
    text: str

@app.post("/summarize")
def summarize(req: SummarizeRequest):
    input_text = req.text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        summary_ids = model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return {"summary": summary}
