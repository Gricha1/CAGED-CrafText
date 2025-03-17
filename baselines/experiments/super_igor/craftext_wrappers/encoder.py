from craftext.craftext_encoder import EncodeForm
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from enum import Enum
import wandb
#from abc import ABC, abstractmethod
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline
from peft import LoraConfig, get_peft_model
from peft import PeftModel
import random
import string

import torch
from baselines.experiments.super_igor.super_dataset import SuperDataset
from baselines.experiments.super_igor.prompts import promt_instruction
from craftext.craftext_encoder import DistilBertEncode
from trl import SFTTrainer
import yaml


from itertools import combinations
import random

def plan_augmentations(plan):
    steps = plan.split("\n")
    combinations_result = []
    for i in range(2, 6):
        combination = combinations(steps, i)
        combination_str = ["\n".join(combo) for combo in list(combination)]
        combinations_result+=combination_str
    combinations_result = list(combinations_result)
    random.shuffle(combinations_result)
    masked_plans = []
    if len(combinations_result[:4])<3:
        masked_plans = ["Deafault_plans Deafault_plans Deafault_plans Deafault_plans Deafault_plans"] * (3-len(combinations_result[:3]))
    return combinations_result[:4]+[plan]+masked_plans
        
def generate_random_word(length):
    letters = string.ascii_lowercase  # используем только строчные буквы для слов
    return ''.join(random.choice(letters) for _ in range(length))

def generate_random_sentence(num_words, word_length):
    words = [generate_random_word(word_length) for _ in range(num_words)]
    # Склеиваем слова в предложение
    sentence = ' '.join(words)
    return sentence.capitalize() + '.'

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
       # print(dataset)
       # exit()
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
      #  model = get_peft_model(model, peft_config)
            model.enable_input_require_grads()
            model = PeftModel.from_pretrained(model, model_name, is_trainable=True).to("cuda")
        
        print("- "*80)
        print()
        print()
        print(model_name)
        print()
        print()
        print("- "*80)
    

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
    

def generate(prompt, model, tokenizer):
    # pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=400)
    # result = pipe(f"{prompt} [/INST]")
    # result = result[0]['generated_text'].split("[/INST]")[1]
   # print(prompt)
    with torch.cuda.amp.autocast():
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output_ids = model.generate(input_ids, max_length=250)
        result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return result

def extract_promt_and_answers(dataset):
    texts = dataset['text']
    prompts = ["Plan:".join(text.split('Plan:')[:2])+"Plan:" for text in texts]
    previos_answes = [text.split('Plan:')[2] for text in texts]
    return {"prompts": prompts, 'previos_plans':previos_answes}
    
def generate_examples(dataset, model, tokenizer):
    dataset_size = len(dataset['text'])
    examples_inices = [random.randint(1, dataset_size) for i in range(5)]
    texts = [dataset['text'][i] for i in examples_inices]
    prompts = ["Plan:".join(text.split('Plan:')[:2])+"Plan:" for text in texts]
    previos_answes = [text.split('Plan:')[2] for text in texts]
    plans = []
    model.eval()
    for prompt in prompts:
        plan = generate(prompt, model, tokenizer)
        plan = plan.split('Plan:')[2] 
        plans.append(plan)
        
    
    predictions = {'prompts':prompts, 'previos_plans': previos_answes, 'current_plans': plans}
    return predictions


class SuperDatasetEncoder(DistilBertEncode):
    def __init__(self, super_dataset, form_to_use=EncodeForm.EMBEDDING, num_return_sequences=5, n_splits=5, augment=False):
        """
        Concrete implementation for Qwen with advanced embedding extraction methods.
        """
        super().__init__(form_to_use=form_to_use, n_splits=n_splits)
        self.super_dataset = super_dataset
        self.augment = augment
        self.DEFAULT_PLAN = "Deafault_plans Deafault_plans Deafault_plans Deafault_plans Deafault_plans"
        self.num_return_sequences = num_return_sequences
    
    def encode(self, instructions, max_new_tokens=100, return_responses=False):
        plans_per_instruction = self._extract_preinited_plans(instructions)
        
        if self.augment:
            augmented_plans = []
            for plan in plans_per_instruction:
                 augmented_plan = plan_augmentations(plan)
                 augmented_plans+=augmented_plan
            plans_per_instruction = augmented_plans
            
        if return_responses:
            return  [super().encode(plans_per_instruction), plans_per_instruction]
        return super().encode(plans_per_instruction)

    def _extract_preinited_plans(self, instructions):
        current_plans = []
        for instruction in instructions:
            print(f"Is {instruction} in SD?")
            print(instruction in self.super_dataset.instructions.keys())
            print(self.super_dataset.instructions.keys())
            if instruction in self.super_dataset.instructions.keys():
                plans = self.super_dataset.instructions[instruction].plan_options
                print(plans)
            else:
                plans = []
              #  exit()
            if len(plans)!=self.num_return_sequences:
                plans += ([self.DEFAULT_PLAN]*(self.num_return_sequences - len(plans)))
                
                print("-----------------")
                print(f"Expectef {self.num_return_sequences} but got {len(plans)}!")
                print("-----------------")
            current_plans.extend(plans)
        print("HELLO WVERYb~ODY!")
        return current_plans
    
from itertools import combinations

class ShapledSuperDatasetEncoder(DistilBertEncode):
    def __init__(self, super_dataset, form_to_use=EncodeForm.EMBEDDING, num_return_sequences=5, n_splits=5):
        """
        Concrete implementation for Qwen with advanced embedding extraction methods.
        """
        super().__init__(form_to_use=form_to_use, n_splits=n_splits)
        self.super_dataset = super_dataset
        self.DEFAULT_PLAN = "Deafault_plans Deafault_plans Deafault_plans Deafault_plans Deafault_plans"
        self.num_return_sequences = num_return_sequences
        
    def augment_plan(self, plan, num_combinations=200):
        """
        Generates a specified number of plan item combinations without changing their order.
        If there are not enough combinations, fills with the original plan.
        If there are too many, shuffles and trims the list.
        
        :param plan: A string with plan items separated by '\n'.
        :param num_combinations: The desired number of combinations.
        :return: A list of lists, where each list is a valid combination of plan items.
        """
        plan_items = plan.split('\n')
        all_combinations = []
        
        for r in range(1, len(plan_items) + 1):
            all_combinations.extend(combinations(plan_items, r))
        
        all_combinations = [list(comb) for comb in all_combinations]
        
        if len(all_combinations) < num_combinations:
            # Fill the remaining slots with the original plan
            while len(all_combinations) < num_combinations:
                all_combinations.append(plan_items)
        else:
            # Shuffle and trim
            random.shuffle(all_combinations)
            all_combinations = all_combinations[:num_combinations]
        
        all_combinations_joined = []
        for combination in all_combinations:
            all_combinations_joined.append("\n".join(combination))
        return all_combinations_joined + [plan]
        
    
    def encode(self, instructions, max_new_tokens=100, return_responses=False):
        plans_per_instruction = self._extract_preinited_plans(instructions)
        augmented_plans = []
        
        for plan in plans_per_instruction:
            a_plan = self.augment_plan(plan)
            #print(a_plan)
            #exit()
            augmented_plans+= a_plan
            
        if return_responses:
            return  [super().encode(augmented_plans), augmented_plans]
        return super().encode(augmented_plans)
    
    
    def _extract_preinited_plans(self, instructions):
        current_plans = []
        for instruction in instructions:
            print(f"Is {instruction} in SD?")
            print(instruction in self.super_dataset.instructions.keys())
            if instruction in self.super_dataset.instructions.keys():
                plans = self.super_dataset.instructions[instruction].plan_options
                print(plans)
            else:
                plans = []
            if len(plans)!=self.num_return_sequences:
                plans += ([self.DEFAULT_PLAN]*(self.num_return_sequences - len(plans)))
                
                print("-----------------")
                print(f"Expectef {self.num_return_sequences} but got {len(plans)}!")
                print("-----------------")
            current_plans.extend(plans)
        print("HELLO WVERYb~ODY!")
        return current_plans
    

#"Qwen/Qwen2.5-3B-Instruct"
class QwenEncodeModel(DistilBertEncode):
    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct", form_to_use=EncodeForm.EMBEDDING, 
                 num_return_sequences=5, load_model=True, n_splits=5, use_expert=True, augment=False):
        """
        Concrete implementation for Qwen with advanced embedding extraction methods.
        augment - use plans augmantaition with steps combination
        """
        super().__init__(form_to_use=form_to_use, n_splits=n_splits)
        base_model_name = "Qwen/Qwen2.5-3B-Instruct"  # Base model name
        self.form_to_use = form_to_use
        self.plan_model_name = model_name
        self.augment = augment
        self.plan_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        self.plan_tokenizer.add_eos_token = True
        self.num_return_sequences = num_return_sequences

        self.plan_model = None  # Initialize the model
        self.lora_model = None  # For LoRA weights if used
        
        self.DEFAULT_PLAN = "Deafault_plans Deafault_plans Deafault_plans Deafault_plans Deafault_plans"
        
        if load_model:
            # Load the base model
            self.plan_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype="auto",  # Automatically use the optimal dtype
                device_map="auto",  # Automatically distribute model layers across devices
                trust_remote_code=True
            ).to("cuda")
            self.plan_model.eval()
            
            # If LoRA weights are used, load them on top of the base model
            if model_name != base_model_name:
                try:
                    
                    # Load LoRA weights
                    self.plan_model = PeftModel.from_pretrained(self.plan_model, self.plan_model_name)
                    self.plan_model.to("cuda")
                    self.plan_model.eval()
                    print("LoRA weights successfully loaded.")
                  #  exit()
                except Exception as e:
                    print(f"Error loading LoRA weights: {e}")
                    exit()
        

    def train(self, dataset, data_for_inference, train_config, full_train_set=None):
        trainer = EncoderTrainer(config=train_config).init_trainer(dataset, self.plan_model_name, self.plan_tokenizer)
        for i in range(2):
            
            examples = generate_examples(data_for_inference, trainer.model, trainer.tokenizer)
            table_bad = wandb.Table(columns=["Prompt", "Previous Plans", "Current Plans"])
            for prompt, prev_plan, curr_plan in zip(examples["prompts"], examples["previos_plans"], examples["current_plans"]):
                table_bad.add_data(prompt, prev_plan, curr_plan)

            examples = generate_examples(dataset, trainer.model, trainer.tokenizer)
            table = wandb.Table(columns=["Prompt", "Previous Plans", "Current Plans"])
            for prompt, prev_plan, curr_plan in zip(examples["prompts"], examples["previos_plans"], examples["current_plans"]):
                table.add_data(prompt, prev_plan, curr_plan)
            
            trainer.train()
            wandb.log({"Generated Examples Good": table})
            wandb.log({"Generated Examples Previosly Bad": table_bad})
            
            table = wandb.Table(columns=["Prompt", "Previous Plans"])
            if full_train_set:
                train_data_dict = extract_promt_and_answers(full_train_set)
                for prompt, prev_plan in zip(train_data_dict["prompts"], train_data_dict["previos_plans"]):
                    table.add_data(prompt, prev_plan)
            wandb.log({"TrainSet": table})
                
            
            trainer.model.save_pretrained(f"{train_config['training_args']['output_dir']}/{i}_")
        

        

    def encode(self, instructions, max_new_tokens=100, return_responses=False):
        """
        Encodes the given prompt directly.

        Parameters:
        - full_prompt: The full text prompt already generated (e.g., using PromptTemplate).
        - max_new_tokens: Maximum number of tokens to generate.

        Returns:
        - Sentence embedding, tokens, or generated response.
        """

        do_plan = False
        if do_plan:
            full_prompt= [f"{promt_instruction(instruction)}" for instruction in instructions]
            
            responses, tokens = self._generate_text(full_prompt, max_new_tokens, num_return_sequences=self.num_return_sequences)
            if self.augment:
                augmented_plans = []
                for plan in responses:
                    augmented_plan = plan_augmentations(plan)
                    augmented_plans+=augmented_plan
                responses = augmented_plans
        else:
            responses = instructions
            
        if return_responses:
            return  [super().encode(responses), responses]
        return super().encode(responses)

    def _generate_text(self, full_prompt, max_new_tokens, num_beams=6, num_return_sequences=5):
        """
        Generates text from the prompt using beam search.

        Parameters:
        - full_prompt: The input prompt for text generation.
        - max_new_tokens: Maximum number of tokens to generate.
        - num_beams: Number of beams for beam search.
        - num_return_sequences: Number of sequences to return.

        Returns:
        - List of generated text responses.
        - List of token IDs corresponding to the generated responses.
        """
        #print(self.plan_model.device)
        
        inputs = self.plan_tokenizer(full_prompt,padding=True, truncation=True, return_tensors="pt").to(self.plan_model.device)
        #exit()
        if num_return_sequences==1:
            num_beam_groups = 2
            num_beams = 2
        else:
            num_beam_groups, num_beams = 10, num_return_sequences
            
        with torch.no_grad():
            if num_return_sequences==1:
                outputs = self.plan_model.generate(
                **inputs,
                max_new_tokens=50,
                eos_token_id= self.plan_tokenizer.eos_token_id,
                num_beams=5,
                do_sample=False,
                early_stopping=True,
                num_return_sequences=num_return_sequences,
                return_dict_in_generate=True
            )
            else:
                        
                outputs = self.plan_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    eos_token_id= self.plan_tokenizer.eos_token_id,
                    num_beams=num_beams,
                    num_beam_groups=num_beam_groups,
                    diversity_penalty=0.2, 
                    do_sample=False,
                    # do_sample=True,
                    # temperature=1,
                    # top_k=50, 
                    # top_p=2.4,  
                    early_stopping=True,
                    num_return_sequences=num_return_sequences,
                    return_dict_in_generate=True
                )
            

        generated_text_ids = outputs.sequences
        responses = self.plan_tokenizer.batch_decode(generated_text_ids, skip_special_tokens=True)

        responses_new =  []
        for r in responses:
            formeted_r = ""
            if "Plan:" in r:
                # print("---"*30)
                # print(r)
                formeted_r = r.split("Plan:")[2]
            elif "1" in r:
                formeted_r = r.split("1")[2]
            else:
                formeted_r = r

            if "Finish!" in r:
                formeted_r = formeted_r.split("Finish!")[0] + "Finish!"
            
            responses_new.append(formeted_r)
        return responses_new, generated_text_ids

  
def SuperEncoder(super_dataset, form_to_use=EncodeForm.EMBEDDING,
                 num_return_sequences=5, n_splits=5, shap = False, augment=False):
    if shap:
        class CustomShapledSuperDatasetEncoder(ShapledSuperDatasetEncoder):
            def __init__(self,  super_dataset=super_dataset, 
                        form_to_use=form_to_use, 
                        num_return_sequences=num_return_sequences, n_splits=n_splits):
                super().__init__(super_dataset=super_dataset, form_to_use=form_to_use,
                                num_return_sequences=num_return_sequences, n_splits=n_splits)
        return CustomShapledSuperDatasetEncoder
    
    class CustomSuperEncoder(SuperDatasetEncoder):
        def __init__(self,  super_dataset=super_dataset, 
                    form_to_use=form_to_use, 
                    num_return_sequences=num_return_sequences, n_splits=n_splits, augment=augment):
            super().__init__(super_dataset=super_dataset, form_to_use=form_to_use,
                            num_return_sequences=num_return_sequences, n_splits=n_splits, augment=augment)
    return CustomSuperEncoder
            

# Фабрика, возвращающая класс с определённым model_name
def QwenModelWrapper(model_name, num_return_sequences,augment=False):
    class CustomQwenEncodeModel(QwenEncodeModel):
        def __init__(self, form_to_use=None, load_model=True, augment=augment):
            super().__init__(model_name=model_name, form_to_use=form_to_use,
                             num_return_sequences=num_return_sequences, load_model=load_model, augment=augment)
    
    return CustomQwenEncodeModel


if __name__ == "__main__":
    # Создаём шаблон промпта
    prompt_template = PromptTemplate("""
    Craftax is a virtual environment designed for exploration, crafting, and task completion. 
    The agent can move, collect resources, craft items, place objects, and interact with its surroundings. 
    Completing tasks often requires the agent to gather resources and craft items before placing them.
    Task: Create a step-by-step action plan (maximum 10 steps) for the agent in Craftax to achieve the following instruction:
    $INSTRUCTION$
    Response Format:
    Provide the plan as a numbered list. Each step should represent a specific action or logical task for the agent to perform, 
    such as gathering resources, crafting items, or placing objects. 
    Keep the steps clear, concise, and focused on fulfilling the instruction efficiently.
    """)

    # Инструкция для задачи
    instruction = "Position the plant centrally and place crafting tables around it."
    
    # Генерация полного промпта
    full_prompt = prompt_template.render(INSTRUCTION=instruction)

    full_prompt = [full_prompt, full_prompt]
    print("Generated Prompt:")
    print(full_prompt)
    print("=" * 80)

    # Используем QwenEncodeModel для различных форм энкодинга
    print("=== Using QwenEncodeModel ===")

    # Инициализация модели для генерации
    model_generation = QwenEncodeModel(form_to_use=EncodeForm.GENERATION)
    generated_response = model_generation.encode(full_prompt)
    print("\nGenerated Response:")
    print(generated_response)
    with open("generated_response.pkl", "wb") as file:
        pickle.dump(generated_response, file)
    print("=" * 80)

    # Инициализация модели для токенов
    model_tokens = QwenEncodeModel(form_to_use=EncodeForm.TOKEN)
    generated_tokens = model_tokens.encode(full_prompt)
    print("\nGenerated Tokens:")
    print(generated_tokens)
    print("=" * 80)

    # Инициализация модели для эмбеддингов
    model_embeddings = QwenEncodeModel(form_to_use=EncodeForm.EMBEDDING)
    generated_embeddings = model_embeddings.encode(full_prompt)
    print("\nGenerated Embeddings:")
    print(generated_embeddings)
    print(generated_embeddings.shape)