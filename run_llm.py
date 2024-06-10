import torch
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto", low_cpu_mem_usage=True)

model.generation_config.cache_implementation = "static"
model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

app = FastAPI()
@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    input_text = data['text']
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs)
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return {"result": result}
