from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

# Load model and tokenizer
model_name = "thantsan/mental_health_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

class Query(BaseModel):
    text: str

@app.post("/predict/")
async def predict(query: Query):
    if not query.text:
        raise HTTPException(status_code=400, detail="Text input is required")
    inputs = tokenizer(query.text, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"])
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
