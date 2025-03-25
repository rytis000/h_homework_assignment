import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
PEFT_MODEL = "./finetuned_mistral_domain_generator"

config = PeftConfig.from_pretrained(PEFT_MODEL)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(PEFT_MODEL)
model = PeftModel.from_pretrained(base_model, PEFT_MODEL)
model.config.use_cache = True


def generate_domain_suggestions(model, tokenizer, business_description, max_length=50):
    instruction = "Generate domain name suggestions for the following business."
    input_text = f"Business description: {business_description}"

    prompt = f"<s>[INST] {instruction}\n\n{input_text} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "[/INST]" in generated_text:
        response = generated_text.split("[/INST]")[1].strip()
    else:
        response = generated_text

    return response


def generate_response(prompt):
    test_description = f"A business that specializes in {prompt}."
    output = generate_domain_suggestions(model, tokenizer, test_description)
    return output.split("Suggested domains: ")[1].split(", ")[:-1]


iface = gr.Interface(fn=generate_response, inputs="text", outputs="text")
iface.launch()
