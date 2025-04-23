from typing import  List
from dataclasses import dataclass
from itertools import combinations
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from baselines.experiments.super_igor.prompts import (
    promt_instruction,
    PlanExtractor,
    PROMPTS
)

DEFAULT_PLAN = "Deafault_plans Deafault_plans Deafault_plans Deafault_plans Deafault_plans"
DEFAULT_STEP = "[MASK]"
    
def plan_augmentations(plan):
    steps = plan.split("\n")
    combinations_result = []
    constraint = 3
    for i in range(2, 6):
        combination = combinations(steps, i)
        combination_str = ["\n".join(combo) for combo in list(combination)]
        combinations_result+=combination_str
    combinations_result = list(combinations_result)
    random.shuffle(combinations_result)
    masked_plans = []
    constrained_combinations = combinations_result[:constraint]+[plan]
    if len(constrained_combinations)<(constraint+1):
        masked_plans = [DEFAULT_PLAN] * ((constraint+1)-len(constrained_combinations))
    return constrained_combinations+masked_plans

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

@dataclass 
class PromptTemplate:
    base = 0
    functions = 1
    fucntions_hints = 2
    
@dataclass
class ModelConfig():
    original_model_path: str
    peft_weights_path: str
    
@dataclass 
class GenerationConfig():
    num_paraphrases: int
    beam_groups: int
    beams_count: int
    max_new_tokens: int
    prompt_template: int
    

class SDPlanner:
    def __init__(self, super_dataset, num_return_sequences: int = 5, augment: bool = False):
        self.super_dataset = super_dataset
        self.num_return_sequences = num_return_sequences
        self.augment = augment
        self.default_plan = DEFAULT_PLAN
    
    def return_plans(self, instructions: List[str]) -> List[str]:
        plans = self.extract_plans(instructions)
        print("Did Plan")
        if self.augment:
            augmented = []
            for plan in plans:
                augmented.extend(plan_augmentations(plan))
            plans = augmented
            print("Augment Plan")
        return plans

    def extract_plans(self, instructions: List[str]) -> List[str]:
        collected = []
        for instruction in instructions:
            if instruction in self.super_dataset.instructions:
                # Extract plans from Super Dataset
                plans = self.super_dataset.instructions[instruction].plan_options
            else:
                plans = []
                
            if len(plans) != self.num_return_sequences:
                # If the number of plans is less than necessary, we add masking plans.
                # This is necessary for correct comparison of plans and instructions in ScenariosLoader
                plans += [self.default_plan] * (self.num_return_sequences - len(plans))
                plans = plans[:self.num_return_sequences]
            

            
            
            collected.extend(plans)
    
        return collected

import gc
class LLMPlanner:
    """
    Generate plans with LLM models
    """
    def __init__(self, model_config: ModelConfig, generation_config: GenerationConfig):
        self.generation_config = generation_config
        self.plan_model, self.plan_tokenizer = self._init_model_and_tokenizer(model_config)
        self.device = self.plan_model.device

    def _init_model_and_tokenizer(self, model_config: ModelConfig):
        base_model = AutoModelForCausalLM.from_pretrained(
            model_config.original_model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_config.original_model_path,
            trust_remote_code=True
        )
        tokenizer.add_eos_token = True

        if (model_config.peft_weights_path != model_config.original_model_path 
            and model_config.peft_weights_path is not None):
            try:
                base_model = PeftModel.from_pretrained(base_model, model_config.peft_weights_path).eval()
            except Exception as e:
                raise(f"Failed to load LoRA weights: {e}")

        return base_model, tokenizer

    def return_plans(self, instructions: list[str]):
        promt_template = PROMPTS[self.generation_config.prompt_template]
        prompts = [f"{promt_instruction(instr, prompt=promt_template)}" for instr in instructions]

        inputs = self.plan_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        if self.generation_config.num_paraphrases != 1:
            num_beams = self.generation_config.beams_count
            num_beam_groups = self.generation_config.beam_groups
        else:
            num_beams = num_beam_groups = 1

        generation_params = {
            "max_new_tokens": self.generation_config.max_new_tokens,
            "eos_token_id": self.plan_tokenizer.eos_token_id,
            "num_beams": num_beams,
            "num_beam_groups": num_beam_groups,
            "diversity_penalty": 0.1 if self.generation_config.num_paraphrases > 1 else 0.0,
            "do_sample": False,
            "early_stopping": True,
            "num_return_sequences": self.generation_config.num_paraphrases,
            "return_dict_in_generate": False
        }

        with torch.no_grad():
            outputs = self.plan_model.generate(**inputs, **generation_params)

        raw_responses = self.plan_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        prompt_type = self.generation_config.prompt_template
        
        formatted = []
        for responce in raw_responses:
            try:
                plan = PlanExtractor.extract(responce,prompt_type)
            except:
                plan = DEFAULT_PLAN
            formatted.append(plan)
        del inputs, outputs 
        torch.cuda.empty_cache() 
        gc.collect() 
        return formatted #, outputs.sequences


def make_planer(planer_type: str, config: dict):
    
    if planer_type == "llm":
        model_cfg = ModelConfig(**config["model_config"])
        generation_cfg = GenerationConfig(**config["generation_config"])
        return LLMPlanner(model_config=model_cfg, generation_config=generation_cfg)

    elif planer_type == "sd":
        return SDPlanner(
            super_dataset=config["super_dataset"],
            num_return_sequences=config["generation_config"]["num_paraphrases"],
            augment=config.get("augment", False)
        )

    else:
        raise ValueError(f"Unknown planner type: {planer_type}")
