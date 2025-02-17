from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftConfig, PeftModel
import torch

def load_peft_model(adapter_path):
    peft_config = PeftConfig.from_pretrained(adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

    # LoRA 학습된 weight를 base model과 합쳐 fine tuned model로 만듦
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        device_map='auto',
        torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        device_map='auto',
        torch_dtype=torch.float16
    )
    model = model.merge_and_unload()

    return model, tokenizer

def create_pipeline(model, tokenizer, max_new_tokens):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens
    )
    return pipe

def generate_response(pipe, prompt):
    formatted_prompt = f"### 질문: {prompt}\n\n### 답변:"
    outputs = pipe(
        formatted_prompt,
        do_sample=True,
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        add_special_tokens=True 
    )
    print(outputs[0]["generated_text"][len(formatted_prompt):])

def main():
    adapter_path = "outputs/checkpoint-50"

    model, tokenizer = load_peft_model(adapter_path)

    pipe_finetuned = create_pipeline(model, tokenizer)

    prompt = "지식인이 뭐야?"

    response = generate_response(pipe_finetuned, prompt)
    print(response)

if __name__ == "__main__":
    main()
