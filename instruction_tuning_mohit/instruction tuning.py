from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
import transformers
import torch

def load_and_preprocess_dataset(dataset_path):
    # 데이터셋을 Hugging Face Dataset 형식으로 변환
    dataset = Dataset.from_json(dataset_path)
    dataset = Dataset.from_dict({
        "input": [item["input"] for item in dataset],
        "output": [item["output"] for item in dataset],
    })

    dataset = dataset.map(
        lambda x: {'text': f"### 질문: {x['input']}\n\n### 답변: {x['output']}<|endoftext|>" }
    )
    dataset = dataset.map(lambda x: tokenizer(x["text"]), batched=True)
    return dataset

def configure_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model_with_quantization(model_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 모델의 가중치를 4비트로 양자화하여 로드
        bnb_4bit_use_double_quant=True,  # 양자화된 값의 정밀도를 높이기 위해 더블 양자화 사용
        bnb_4bit_quant_type="nf4",  # 4비트 양자화에 사용되는 특정 타입
        bnb_4bit_compute_dtype=torch.bfloat16   
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, bnb_config, device_map={"":0})
    model = prepare_model_for_kbit_training(model)
    return model

def load_model(model_id):
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"":0})
    model = prepare_model_for_kbit_training(model)
    return model

def apply_lora_configuration(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 인과적 언어 모델링(Causal Language Modeling)
        inference_mode=False,  # 추론 모드 : False는 학습
        r=8,  # LoRA의 저랭크 행렬의 크기
        lora_alpha=32,  # LoRA 알파 값 : LoRA 적용 시 기존 가중치 행렬에 곱해지는 스케일 팩터
        lora_dropout=0.2,  # LoRA 적용 후 드롭아웃
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

def create_trainer(model, dataset, tokenizer):
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        fp16=False,
        logging_steps=10,
        output_dir="instruction_tuning_mohit/cynical_outputs"
    )

    data_collator=transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,  # 사용할 토크나이저를 지정
        mlm=False  # 마스크된 언어 모델링 사용하지 않음
    )

    trainer = transformers.Trainer(
        model=model,  # 학습할 모델
        train_dataset=dataset,  # 학습 데이터셋
        args=training_args,
        data_collator=data_collator,
    )
    return trainer

def main():
    dataset_path = "instruction_tuning_mohit/cynical_dataset.json"
    model_id = "Bllossom/llama-3.2-Korean-Bllossom-3B"

    global tokenizer
    tokenizer = configure_tokenizer(model_id)

    dataset = load_and_preprocess_dataset(dataset_path)

    model = load_model(model_id)
    model = apply_lora_configuration(model)

    trainer = create_trainer(model, dataset, tokenizer)
    trainer.train()

if __name__ == "__main__":
    main()