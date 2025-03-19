import pickle
import random
import string

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from craftext.craftext_encoder import EncodeForm, DistilBertEncode

from baselines.experiments.super_igor.super_dataset import SuperDataset
from baselines.experiments.super_igor.prompts import promt_instruction
from baselines.experiments.super_igor.craftext_wrappers.encoder_trainer import EncoderTrainer


from itertools import combinations
import random

DEFAULT_PLAN = "Deafault_plans Deafault_plans Deafault_plans Deafault_plans Deafault_plans"
DEFAULT_STEP = "[MASK]"
# --------------- Utils --------------- 

def split_plans(plan, max_steps_length=16):
    steps = plan.split("\n")
    if len(steps)<max_steps_length:
        steps += [DEFAULT_STEP]*(max_steps_length -  len(steps))
    else:
        steps = steps[:max_steps_length]
    return steps
    
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
        masked_plans = [DEFAULT_PLAN] * (3-len(combinations_result[:3]))
    return combinations_result[:4]+[plan]+masked_plans
        
def generate_random_word(length):
    letters = string.ascii_lowercase 
    return ''.join(random.choice(letters) for _ in range(length))

def generate_random_sentence(num_words, word_length):
    words = [generate_random_word(word_length) for _ in range(num_words)]
    sentence = ' '.join(words)
    return sentence.capitalize() + '.'

def generate(prompt, model, tokenizer):
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

# --------------- Encoder - Load plans from sd --------------- 
class SuperDatasetEncoder(DistilBertEncode):
    def __init__(self, super_dataset, form_to_use=EncodeForm.EMBEDDING, 
                 num_return_sequences=5, n_splits=1, augment=False, split_into_steps=False):
        """
        Concrete implementation for Qwen with advanced embedding extraction methods.
        """
        super().__init__(form_to_use=form_to_use, n_splits=n_splits)
        self.super_dataset = super_dataset
        self.augment = augment
        self.DEFAULT_PLAN = DEFAULT_PLAN
        self.num_return_sequences = num_return_sequences
        self.split_into_steps = split_into_steps
    
    def encode(self, instructions, max_new_tokens=100, return_responses=False):
        plans_per_instruction = self._extract_preinited_plans(instructions)
        if self.augment:
            augmented_plans = []
            for plan in plans_per_instruction:
                 augmented_plan = plan_augmentations(plan)
                 augmented_plans+=augmented_plan
            plans_per_instruction = augmented_plans
        
        if self.split_into_steps:
            plans_per_instruction = [split_plans(plan) for plan in plans_per_instruction]
            embeddings_list = [super().encode(steps) for steps in plans_per_instruction]
        else: 
            embeddings_list = super().encode(plans_per_instruction)
        if return_responses:
            return  [embeddings_list, plans_per_instruction]
        return embeddings_list

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
            if len(plans)!=self.num_return_sequences:
                plans += ([self.DEFAULT_PLAN]*(self.num_return_sequences - len(plans)))
                print("-----------------")
                print(f"Expectef {self.num_return_sequences} but got {len(plans)}!")
                print("-----------------")
            current_plans.extend(plans)
        print("HELLO WVERYb~ODY!")
        return current_plans
    
# --------------- Encoder - Predict plans with LLM --------------- 
from baselines.experiments.super_igor.craftext_wrappers.onehot_encoder import encode_plans

def format_responces(responses):
        responses_new =  []
        for r in responses:
            formeted_r = ""
            if "Plan:" in r:
                formeted_r = r.split("Plan:")[2]
            elif "1" in r:
                formeted_r = r.split("1")[2]
            else:
                formeted_r = r
            if "Finish!" in r:
                formeted_r = formeted_r.split("Finish!")[0] + "Finish!"
            responses_new.append(formeted_r)
        return responses_new
    
class QwenEncodeModel(DistilBertEncode):
    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct", form_to_use=EncodeForm.EMBEDDING, 
                 num_return_sequences=5, load_model=True, n_splits=1, use_expert=True,
                 augment=False, split_into_steps=False, do_plan=True, make_one_hot=False):
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
        self.split_into_steps = split_into_steps
        self.do_plan = do_plan
        self.make_one_hot=make_one_hot

        self.plan_model = None  # Initialize the model
        self.lora_model = None  # For LoRA weights if used
        
        self.DEFAULT_PLAN = DEFAULT_PLAN
        
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

        #self.do_plan = False
        if self.do_plan:
            full_prompt= [f"{promt_instruction(instruction)}" for instruction in instructions]
            
            plans_per_instruction, tokens = self._generate_text(full_prompt, max_new_tokens, num_return_sequences=self.num_return_sequences)
            print(plans_per_instruction)
            if self.augment:
                augmented_plans = []
                for plan in plans_per_instruction:
                    augmented_plan = plan_augmentations(plan)
                    augmented_plans+=augmented_plan
                plans_per_instruction = augmented_plans
    
        else:
            plans_per_instruction = instructions
        
        plans_original = plans_per_instruction
        
        if self.make_one_hot:
            encode_f = encode_plans
        else:
            encode_f = super().encode
            
        if self.split_into_steps:
            plans_per_instruction = [split_plans(plan) for plan in plans_per_instruction]
          #  print(len(plans_per_instruction[0]))
            
            embeddings_list = []
            for plan in plans_per_instruction:
                #print(len(plan))
                embedings = []
                for step in plan:
                    emb = encode_f([step])[0]
                    print(emb.shape)
                    embedings.append(emb)
                embeddings_list.append(embedings)
        else: 
            embeddings_list = encode_f(plans_per_instruction)

        plans_per_instruction = plans_original
        
        if return_responses:
            return  [embeddings_list, plans_per_instruction]
        return embeddings_list

    def _generate_text(self, full_prompt, max_new_tokens=50, num_beams=5, num_return_sequences=5):
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
        inputs = self.plan_tokenizer(
            full_prompt, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.plan_model.device)

        if num_return_sequences!=1:
            num_beams, num_beam_groups = num_return_sequences, num_return_sequences
        else: 
            num_beams, num_beam_groups = 1, 1

        generation_params = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": self.plan_tokenizer.eos_token_id,
            "num_beams": num_beams,
            "num_beam_groups": num_beam_groups,
            "diversity_penalty": 0.1 if num_return_sequences > 1 else 0.0, 
            "do_sample": False,
            "early_stopping": True,
            "num_return_sequences": num_return_sequences,
            "return_dict_in_generate": True
        }

        with torch.no_grad():
            outputs = self.plan_model.generate(**inputs, **generation_params)

        responses = self.plan_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        responses_new = format_responces(responses)
        
        return responses_new, outputs.sequences



# --------------- Encoder - SHAP --------------- 
from itertools import combinations
class ShapledSuperDatasetEncoder(DistilBertEncode):
    def __init__(self, super_dataset, form_to_use=EncodeForm.EMBEDDING, num_return_sequences=5, n_splits=1):
        """
        Concrete implementation for Qwen with advanced embedding extraction methods.
        """
        super().__init__(form_to_use=form_to_use, n_splits=n_splits)
        self.super_dataset = super_dataset
        self.DEFAULT_PLAN = DEFAULT_PLAN
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
            while len(all_combinations) < num_combinations:
                all_combinations.append(plan_items)
        else:
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
    
# --------------- Encoder - Fabrics --------------- 
def SuperEncoder(super_dataset, form_to_use=EncodeForm.EMBEDDING,
                 num_return_sequences=5, n_splits=1, shap = False, augment=False, split_into_steps=False):
    if split_into_steps:
        n_splits=1
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
                    num_return_sequences=num_return_sequences, n_splits=n_splits, 
                    augment=augment, split_into_steps=split_into_steps):
            super().__init__(super_dataset=super_dataset, form_to_use=form_to_use,
                            num_return_sequences=num_return_sequences, n_splits=n_splits, 
                            augment=augment, split_into_steps=split_into_steps)
    return CustomSuperEncoder
            

# Фабрика, возвращающая класс с определённым model_name
def QwenModelWrapper(model_name, num_return_sequences,augment=False, split_into_steps=False, do_plan=True, make_one_hot=False):
    if split_into_steps:
        n_splits=1
    class CustomQwenEncodeModel(QwenEncodeModel):
        def __init__(self, form_to_use=None, load_model=True, augment=augment,split_into_steps=split_into_steps, 
                     n_splits=n_splits, do_plan=do_plan):
            super().__init__(model_name=model_name, form_to_use=form_to_use,n_splits=n_splits,
                             num_return_sequences=num_return_sequences, load_model=load_model,
                             augment=augment,split_into_steps=split_into_steps, do_plan=do_plan, make_one_hot=make_one_hot)
    
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