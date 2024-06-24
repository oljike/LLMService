import os

os.environ["WANDB_PROJECT"] = "PeftExamples"
import transformers
from peft import (
    LoraConfig,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,

)
from datasets import load_dataset, Dataset

if __name__=="__main__":


    model_name = "ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16"
    tokenizer_iqlm = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype="auto",
    )
    model.resize_token_embeddings(len(tokenizer_iqlm))

    # ADD lora
    config = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.0, target_modules=["embed_tokens", "lm_head", "q_proj", "v_proj"]
    )
    model = get_peft_model(model, config)
    print(model.print_trainable_parameters())
    print(model)


    # preprocess data
    raw_data = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction")
    def get_code(s):
        s = s.split("```")
        for p in s:
            if p.startswith('python'):
                return p[6:]


    new_data = []
    for x in raw_data['train']:

        curr_chat = []
        code = get_code(x['answer'])
        if code is None or code == "None":
            continue

        chat = [
            {"role": "user", "content": x["query"]},
            {"role": "assistant", "content": str({"code": code})}]

        new_data.append(chat)

    tokenizer_iqlm.pad_token = tokenizer_iqlm.eos_token
    train_dataset2 = Dataset.from_dict({"chat": new_data})

    train_dataset2 = train_dataset2.map(
        lambda x: {"input_ids": tokenizer_iqlm.apply_chat_template(x["chat"], tokenize=True,
                                                                   truncation=True, padding=True, max_length=512)},
        batched=True)

    train_dataset2 = train_dataset2.remove_columns([col for col in train_dataset2.column_names if col != "input_ids"])
    train_dataset2.set_format("torch")

    training_args = TrainingArguments(
        output_dir="mistral_lora_clm_with_added_tokens",
        num_train_epochs=2,
        save_total_limit=5,
        per_device_train_batch_size=8,
        warmup_steps=10,
        weight_decay=0.0001,
        dataloader_drop_last=True,
        fp16=True,
        logging_steps=10,
        learning_rate=1e-5,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset2,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer_iqlm, mlm=False),

    )
    model.config.use_cache = False
    trainer.train()