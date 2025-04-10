import pickle

from dataclasses import dataclass

from craftext.craftext_encoder import EncodeForm, DistilBertEncode

from baselines.experiments.super_igor.craftext_wrappers.encoder_trainer import EncoderTrainer
from baselines.experiments.super_igor.craftext_wrappers.planners import SDPlanner, LLMPlanner, DEFAULT_STEP
from baselines.experiments.super_igor.craftext_wrappers.onehot_encoder import encode_plans
from baselines.experiments.super_igor.craftext_wrappers.planners import make_planer


import random

# --------------- Utils --------------- 
def split_plans(plan, max_steps_length=16):
    steps = plan.split("\n")
    if len(steps)<max_steps_length:
        steps += [DEFAULT_STEP]*(max_steps_length -  len(steps))
    else:
        steps = steps[:max_steps_length]
    return steps
# --------------- Encoder - Load plans from sd --------------- 

from enum import Enum

@dataclass
class EmbeddingSource(Enum):
    bert=0
    onehot=1
    
class EncoderWithPlanning(DistilBertEncode):
    def __init__(self, planer, encoder=EmbeddingSource.bert,
                 form_to_use=EncodeForm.EMBEDDING, step_by_step = True):
        super().__init__(form_to_use=form_to_use, n_splits=1)
        self.planer = planer
        self.embedding_source = encoder
        # Mode for giving plan step-by-step
        self.step_by_step = step_by_step
    
    def encode(self, instruction, return_responses=False):
        plans = self.planer.return_plans(instruction)
        if self.embedding_source == EmbeddingSource.onehot.value:
            encode_f = encode_plans
        else:
            encode_f = super().encode
        
        # Encode each step
        if self.step_by_step:
            plans_per_instruction = [split_plans(plan) for plan in plans]
            embeddings_list = []
            for plan in plans_per_instruction:
                embedings = []
                for step in plan:
                    emb = encode_f([step])[0]
                    embedings.append(emb)
                embeddings_list.append(embedings)
        else: 
            embeddings_list = encode_f(plans)
        
        if return_responses:
            return [embeddings_list, plans]
        return embeddings_list

def make_encoder_with_planning(
    planer_type: str,
    planer_config: dict,
    embedding_source: int = EmbeddingSource.bert,
    step_by_step: bool = True
) -> type[EncoderWithPlanning]:
    """
    Factory that returns a subclass of EncoderWithPlanning preconfigured
    with planner and encoder parameters.

    Returns:
        A class inheriting from EncoderWithPlanning
    """

    planer = make_planer(planer_type, planer_config)

    class CustomEncoderWithPlanning(EncoderWithPlanning):
        def __init__(self, *args, **kwargs):
            super().__init__(
                planer=planer,
                encoder=embedding_source,
                step_by_step=step_by_step,
                *args,
                **kwargs
            )

    return CustomEncoderWithPlanning

def make_encoder_with_planning(
    planer_type: str,
    planer_config: dict,
    embedding_source: EmbeddingSource = EmbeddingSource.bert,
    step_by_step: bool = True
) -> type[EncoderWithPlanning]:
    """
    Factory that returns a subclass of EncoderWithPlanning
    preconfigured with planner and encoder setup.

    Returns:
        A class inheriting from EncoderWithPlanning.
    """

    planer = make_planer(planer_type, planer_config)

    class CustomEncoderWithPlanning(EncoderWithPlanning):
        def __init__(self,form_to_use,
                     planer=planer,
                     encoder=embedding_source,
                     step_by_step=step_by_step ):
            super().__init__(
                planer=planer,
                encoder=encoder,
                step_by_step=step_by_step,
                form_to_use=form_to_use
            )

    return CustomEncoderWithPlanning


if __name__ == "__main__":
    encoder = make_encoder_with_planning(
        planer_type="llm",
        planer_config={
            "model_config": {
                "original_model_path": "Qwen/Qwen2.5-3B-Instruct",
                "peft_weights_path": None
            },
            "generation_config": {
                "num_paraphrases": 5,
                "augment": True,
                "beam_groups": 2,
                "beams_count": 4,
                "max_new_tokens": 128,
                "prompt_template": 1
            },
            "super_dataset": None,
            "augment": False
        },
        embedding_source=EmbeddingSource.bert,
        step_by_step=True
    )
