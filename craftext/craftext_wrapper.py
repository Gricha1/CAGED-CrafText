# Standard libraries
import os
import random
import importlib
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Any
import torch
# Libraries for JAX and NumPy operations
import jax
import jax.numpy as jnp
from jax import lax, vmap
import numpy as np

# Frameworks and models
from gym import Wrapper
from flax import linen as nn
from transformers import FlaxBertModel, BertTokenizer

# External distribution libraries
import distrax

# Local project imports
from craftext.checkers.base_functions.state_adapter_craftax_classic import GameDataClassic
from craftext.scenarios_loader import load_scenarios, parse_craftext_settings

from transformers import AutoTokenizer, AutoModel

from flax import struct
from sentence_transformers import SentenceTransformer

import os
os.environ['HF_HOME'] = "."

@dataclass
class ScenarioData:
    instructions_list: list
    checkers_list: list
    indices_list: list
    encoded_instructions_list: list
    embeddings_list: list

@dataclass
class ScenarioDataJAX:
    encoded_instructions_list: jnp.array
    embeddings_list: jnp.array
    checkers_list: list

def create_game_data_for_all(state_vec, action_vec):
    return jax.vmap(GameData.from_state)(state_vec, action_vec)
    
@struct.dataclass
class TextEnvState:
    env_state: Any
    timestep: int
    instruction: Optional[jnp.ndarray] 
    idx: int

def get_configs_path():
    module_path = craftext.__spec__.submodule_search_locations[0]
    return os.path.join(module_path, 'configs')

class InstructionWrapper(Wrapper):
    model_name: str = "distilbert-base-uncased"

    def __init__(self, env):
        super().__init__(env)
        self.tokenizer = self._initialize_tokenizer()
        self.model = self._initialize_model()
        self.instruction_str = "None"
        self.encoded_instruction =  self._get_embeddings(self.instruction_str) #jnp.array([[0]])
        
        self.all_scenario = self._load_scenarios()
        self.scenario_data = self._prepare_scenarios()
        self.scenario_data_jax = self._prepare_jax_scenarios()

        self.env = env
        
        print("Init Instruction Wrapper")

    def _initialize_model(self):
        return AutoModel.from_pretrained(self.model_name, cache_dir=".")

    def _initialize_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name, cache_dir=".")
    
    def _get_embeddings(self, instruction):
        inputs = self.tokenizer(instruction, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Get embeddings from CLS tokem
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.numpy()

    def _encode_instruction(self, instruction):
        return self.tokenizer(instruction, padding=True, truncation=True, return_tensors='np')['input_ids']

    def _load_scenarios(self):
        return load_scenarios()

    def _prepare_scenarios(self):
        """Prepares the scenarios data and tokenizes the instructions."""
        instructions_list = []
        checkers_list = []
        indices_list = []
        encoded_instructions_list = []
        embeddings_list = []

        mode, instruction_type, _ = parse_craftext_settings()

        for idx, (key, scenario) in enumerate(self.all_scenario.items()):
            instructions = [scenario['instruction']]
            checkers = [scenario['check_lambda']]

            if instruction_type == 'instruction_with_paraphrases' and 'instruction_paraphrases' in scenario:
                instructions += scenario['instruction_paraphrases']
                checkers += [scenario['check_lambda']] * len(scenario['instruction_paraphrases'])

            for instruction, checker in zip(instructions, checkers):
                encoded_instruction = self._encode_instruction(instruction)
                embeddings = self._get_embeddings(instruction)

                instructions_list.append(instruction)
                checkers_list.append(checker)
                indices_list.append(idx)
                encoded_instructions_list.append(encoded_instruction)
                embeddings_list.append(embeddings)

        
        return ScenarioData(
            instructions_list=instructions_list,
            checkers_list=checkers_list,
            indices_list=np.array(indices_list).reshape(len(instructions_list), 1),
            encoded_instructions_list=np.array(embeddings_list).reshape(len(instructions_list), -1),
            embeddings_list=np.array(embeddings_list).reshape(len(instructions_list), -1)
        )

    def _prepare_jax_scenarios(self):
        encoded_instructions_jax = jnp.array(self.scenario_data.encoded_instructions_list)
        embeddings_jax = jnp.array(self.scenario_data.embeddings_list)
        checkers_list_jax = self._prepare_jax_checkers(self.scenario_data.checkers_list)
        
        return ScenarioDataJAX(
            encoded_instructions_list=encoded_instructions_jax,
            embeddings_list=embeddings_jax,
            checkers_list=checkers_list_jax
        )

    def _prepare_jax_checkers(self, checkers_list):
        indices = jnp.arange(len(checkers_list)) 
        def apply_checker(i, x, y):
            return lax.switch(i, checkers_list, x, y)
        vmap_checkers = vmap(apply_checker, in_axes=(None, None, None))
        return vmap_checkers
        

    def reset(self, _rng, env_params):
        obs, state = self.env.reset(_rng, env_params)
        idx = jax.random.randint(_rng, shape=(), minval=0, maxval=len(self.scenario_data_jax.embeddings_list))
        instructions_emb = self.scenario_data_jax.embeddings_list[idx]
        
        state = TextEnvState( env_state = state, 
                              timestep = state.timestep,
                              instruction = instructions_emb,
                              idx=idx )
        
        return obs, state
        
    def step(self, _rng, env_state, action, env_params):

        idx = env_state.idx
        instructions_emb = env_state.instruction
        
        obs, state, reward, done, info = self.env.step(_rng, env_state.env_state, action, env_params)
        

        game_data_vector = GameDataClassic.from_state(state, action)

        instruction_done = lax.switch(idx,  self.scenario_data.checkers_list, game_data_vector, idx) 
        reward /= 50  
        def update_reward(instruction_done_elem, reward_elem):
            return jax.lax.cond(instruction_done_elem, 
                                lambda _: reward_elem + 1, 
                                lambda _: reward_elem, 
                                operand=None)
            
        reward = update_reward(instruction_done, reward)
        done = instruction_done | done
        
        state = TextEnvState(env_state = state, 
                             timestep = state.timestep,
                             instruction = instructions_emb,
                              idx = idx)
        
        return obs, state, reward, done, info


if __name__=="__main__":
    instruction, checker = get_random_instruction_and_checker()
    print(instruction)