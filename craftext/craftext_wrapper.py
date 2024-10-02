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
from craftext.checkers.base_functions.state_adapter import GameData
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
class TestTextEnvState:
    env_state: Any
    instruction: Optional[jnp.ndarray] 
    idx: int
    
@struct.dataclass
class TextEnvState:
    env_state: Any
    timestep: int
    instruction: Optional[jnp.ndarray] 
    idx: int
   
class InstructionWrapper(Wrapper):
    model_name: str = "distilbert-base-uncased"

    def __init__(self, env, num_envs):
        super().__init__(env)
        self.tokenizer = self._initialize_tokenizer()
        self.model = self._initialize_model()
        self.instruction_str = "None"
        self.encoded_instruction =  self._get_embeddings(self.instruction_str) #jnp.array([[0]])
        
        self.all_scenario = self._load_scenarios()
        self.scenario_data = self._prepare_scenarios()
        self.scenario_data_jax = self._prepare_jax_scenarios()

        self.env = env
        self.num_envs = num_envs
        
        print("Init Instruction Wrapper")

    def init_instructions_ix(self,rng_key):
        def select_instruction_ix(rng, n_instructions):
            return jax.random.randint(rng, shape=(), minval=0, maxval=n_instructions)

        instructions_vmap = jax.vmap(select_instruction_ix, (0, None))
        rngs = jax.random.split(rng_key,  self.num_envs)
        return instructions_vmap(rngs, len(self.all_scenario))

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


    def inf_reset(self, _rng, env_params):
        """
        Used for inference in view_ppo_agent
        """
        obs, state = self.env.reset(_rng, env_params)
        instructions_indices = self.init_instructions_ix(_rng)
        instructions_emb = self.scenario_data_jax.embeddings_list[instructions_indices]
        state = TestTextEnvState(env_state = state, 
                             instruction = instructions_emb,
                             idx = instructions_indices)
        return obs, state
        

    def reset(self, _rng, env_params):
        obs, state = self.env.reset(_rng, env_params)
        instructions_indices = self.init_instructions_ix(_rng)
        idx = jax.random.randint(_rng, shape=(), minval=0, maxval=len(self.scenario_data_jax.embeddings_list))
        instructions_emb = self.scenario_data_jax.embeddings_list[idx]
        
        state = TextEnvState( env_state = state, 
                              timestep = state.timestep,
                              instruction = instructions_emb,
                              idx=idx )
        
        return obs, state

    def inf_step(self, _rng, env_state, action, env_params):
        """
        Used for inference in view_ppo_agent 
        """
        instructions_emb = env_state.instruction
        instructions_indices = env_state.idx 
        
        obs, state, reward, done, info = self.env.step(_rng, env_state.env_state, action, env_params)
        game_data_vector = GameData.from_state(state, action)
        
        instruction_done = self.scenario_data.checkers_list[0]( game_data_vector, 0)
        done = instruction_done | done
        state = TestTextEnvState(env_state = state, 
                            instruction = instructions_emb,
                            idx = instructions_indices)
        
        return obs, state, reward, done, info
        
    def step(self, _rng, env_state, action, env_params):

        idx = env_state.idx
        instructions_emb = env_state.instruction
        
        obs, state, reward, done, info = self.env.step(_rng, env_state.env_state, action, env_params)
        

        game_data_vector = GameData.from_state(state, action)

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