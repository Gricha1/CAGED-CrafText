from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, pipeline
from peft import LoraConfig
from peft import PeftModel
import torch
from trl import SFTTrainer



class EncoderTrainer():
    def __init__(self, config):

        self.config = config 
        self.training_config = self.config['training_args']
        self.lora_config = self.config['q_lora']
        self.bnb_config = self.configure_bnb_config(**self.config['bitsandbytes'])
    
    def configure_bnb_config(self, use_4bit, bnb_4bit_quant_type, bnb_4bit_compute_dtype, use_nested_quant):
        """
        Configure BitsAndBytes for model quantization.
        """
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        return BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_nested_quant,
        )

    def init_trainer(self, dataset, model_name, tokenizer):
        """
        Initialize the SFT Trainer with the given configuration.
        """
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            quantization_config=self.bnb_config,
            device_map=self.config['device_map']
        )

        model.config.use_cache = False
        model.config.pretraining_tp = 1

        peft_config = LoraConfig(
            lora_alpha=self.lora_config['lora_alpha'],
            lora_dropout=self.lora_config['lora_dropout'],
            r= self.lora_config['lora_r'],
            bias= "none",
            task_type="CAUSAL_LM",
        )

        if not "Qwen" in model_name:
            model.enable_input_require_grads()
            model = PeftModel.from_pretrained(model, model_name, is_trainable=True).to("cuda")

        training_arguments = TrainingArguments(
            output_dir=self.training_config['output_dir'],
            num_train_epochs=int(self.training_config['num_train_epochs']),
            per_device_train_batch_size=int(self.training_config['per_device_train_batch_size']),
            per_device_eval_batch_size=int(self.training_config['per_device_eval_batch_size']),
            gradient_accumulation_steps=int(self.training_config['gradient_accumulation_steps']),
            gradient_checkpointing=bool(self.training_config['gradient_checkpointing']),
            optim=str(self.training_config['optim']),
            save_steps=int(self.training_config['save_steps']),
            logging_steps=int(self.training_config['logging_steps']),
            learning_rate=float(self.training_config['learning_rate']),
            weight_decay=float(self.training_config['weight_decay']),
            fp16=bool(self.training_config['fp16']),
            bf16=bool(self.training_config['bf16']),
            max_grad_norm=float(self.training_config['max_grad_norm']),
            max_steps=int(self.training_config['max_steps']),
            warmup_ratio=float(self.training_config['warmup_ratio']),
            group_by_length=bool(self.training_config['group_by_length']),
            lr_scheduler_type=str(self.training_config['lr_scheduler_type']),
            report_to="wandb"
        )


        return SFTTrainer(
            model=model,
            train_dataset=dataset,#['train'],
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=None, #TODO: How to load None from configs?
            tokenizer=tokenizer,
            args=training_arguments,
            packing=self.config['sft']['packing'],
        )
    