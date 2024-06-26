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
    code_dataset = Dataset.from_dict({"chat": new_data[:100]})

    code_dataset = code_dataset.map(
        lambda x: {"chat_format": tokenizer_iqlm.apply_chat_template(x["chat"], tokenize=False)},
        batched=True)

    text_column = "chat_format"
    max_length = 512
    tokenizer = tokenizer_iqlm


    def preprocess_function(examples):
        batch_size = len(examples[text_column])

        # tokenize the user query
        model_inputs = tokenizer(examples[text_column])
        print(model_inputs.keys())

        for i in range(batch_size):
            # get input and answers
            sample_input_ids = model_inputs["input_ids"][i]

            # print(i, sample_input_ids, label_input_ids)
            # concatenate the input with the answer
            model_inputs["input_ids"][i] = sample_input_ids

            # attention is set to 1 for all input
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

        # work on padding all seqs
        for i in range(batch_size):
            # get input and answers
            sample_input_ids = model_inputs["input_ids"][i]

            # padd the whole input with pad_id from the left of the seq
            model_inputs["input_ids"][i] = sample_input_ids + [tokenizer.pad_token_id] * (
                    max_length - len(sample_input_ids))

            # apply the padding to the attention mask also
            model_inputs["attention_mask"][i] = model_inputs[
                                                    "attention_mask"
                                                ][i] + [0] * (max_length - len(sample_input_ids))

            model_inputs["input_ids"][i] = model_inputs["input_ids"][i][:max_length]
            model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i][:max_length]

        return model_inputs


    code_dataset_split = code_dataset.train_test_split(0.2)

    processed_datasets = code_dataset_split.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=code_dataset_split['train'].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]

    tokenizer_iqlm.pad_token = tokenizer_iqlm.eos_token
    training_args = TrainingArguments(
        output_dir="mistral_lora_clm_with_added_tokens",
        num_train_epochs=2,
        save_total_limit=5,
        per_device_train_batch_size=2,
        warmup_steps=10,
        weight_decay=0.0001,
        dataloader_drop_last=True,
        fp16=True,
        logging_steps=10,
        learning_rate=1e-5,
        # gradient_checkpointing=True,
        # gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer_iqlm, mlm=False),
        # data_collator = transformers.default_data_collator

    )
    model.config.use_cache = False
    trainer.train()

    trainer.push_to_hub(token="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")
    trainer.model.push_to_hub(training_args.output_dir)