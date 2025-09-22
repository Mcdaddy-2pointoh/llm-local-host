# Imports
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import yaml
from dotenv import dotenv_values
import os

# Basic Initialisation of server
app = FastAPI(title="Local HuggingFace based Small Language Model API", version="1.0")

# Pydantic class for input validation
class Request(BaseModel):
    prompt: str

# Load HF token
env_config = dotenv_values(".env")

# Load the Model configs
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)


# Set the HF token key in environ & HF_HOME path
os.environ['HF_TOKEN'] = env_config['HF_KEY']
os.environ['HF_HOME'] = config['hf_home_path']

model_name = config['hf_model']['model_name']
device_map = config['hf_model']['device_map']
torch_dtype = config['hf_model']['torch_dtype']
quantization = config['hf_model']['quantization']

# Load Quantization config
if quantization is not None and quantization['quantization_type'] == 'bnb':
    if quantization['load_in_4_bit']:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)

# Load model tokenizers
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Torch Dtypes
torch_dtype_dict = {
    "torch.float16": torch.bfloat16
}

# Load CausalLM
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    # dtype=torch_dtype_dict[torch_dtype],
    quantization_config=bnb_config
)

# Server host code
@app.post("/generate")
def generate(req: Request):
    inputs = tokenizer(req.prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=750)
    return {"response": tokenizer.decode(outputs[0], skip_special_tokens=True)}