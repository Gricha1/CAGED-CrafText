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
from craftext.checkers.base_functions.state_adapter_craftax_classic import GameDataClassic
from craftext.scenarios_loader import load_scenarios, parse_craftext_settings, load_config_or_env, get_configs_path

from transformers import AutoTokenizer, AutoModel

from flax import struct
from sentence_transformers import SentenceTransformer
import yaml

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
    success_rate: float
    total_success_rate: float
    environment_key: int
    


class InstructionWrapper(Wrapper):
    model_name: str = "distilbert-base-uncased"

    def __init__(self, env, dataset_configuration=None):
        super().__init__(env)
        self.tokenizer = self._initialize_tokenizer()
        self.model = self._initialize_model()
        self.instruction_str = "None"
        self.encoded_instruction =  self._get_embeddings(self.instruction_str) #jnp.array([[0]])
        
        _, instruction_type, _, _, _ = self._load_config(dataset_configuration)
        use_parafrases = instruction_type == 'instruction_paraphrases'
        self.all_scenario, self.environment_key = self._load_scenarios(dataset_configuration)
        self.scenario_data = self._prepare_scenarios(use_parafrases)
        self.scenario_data_jax = self._prepare_jax_scenarios()
        
        if self.environment_key == 1:
            self.StateStructure = GameData
        else: 
            self.StateStructure = GameDataClassic
            print("Use Classsic State adapter")

        self.env = env
        print(" ----------------------- ")
        print(self.environment_key)
        
        print("Init Instruction Wrapper")
    
    
    def _load_config(self, config_name):
        if config_name is None:
            return None
        config_path = get_configs_path()
        config_name = str(os.path.join(config_path, config_name))+".yaml"
        config = load_config_or_env(config_name)
        return config
    
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

    def _load_scenarios(self, config_name):
        return load_scenarios(config_name)

    def _prepare_scenarios(self, use_parafrases):
        """Prepares the scenarios data and tokenizes the instructions."""
        instructions_list = []
        checkers_list = []
        indices_list = []
        encoded_instructions_list = []
        embeddings_list = []

        #mode, instruction_type, _ = parse_craftext_settings()

        for idx, (key, scenario) in enumerate(self.all_scenario.items()):
            if 'instruction' in list(scenario.keys()):
                instructions = [scenario['instruction']]
                checkers = [scenario['check_lambda']]
            else:
                instructions = []
                checkers = []

            if use_parafrases and 'instruction_paraphrases' in scenario:
                instructions += scenario['instruction_paraphrases']
                checkers += [scenario['check_lambda']] * len(scenario['instruction_paraphrases'])
                print("YEP!")
                print("YEP!")
                print("YEP!")

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
                              idx=idx,
                              environment_key = self.environment_key,
                              success_rate = 0.,
                              total_success_rate = 0.)
        

        return obs, state
        
    def step(self, _rng, env_state, action, env_params):
        idx = env_state.idx
        instructions_emb = env_state.instruction
        environment_key = env_state.environment_key
        
        obs, state, reward, done, info = self.env.step(_rng, env_state.env_state, action, env_params)
        
        
        game_data_vector = self.StateStructure.from_state(state, action)

        instruction_done = lax.switch(idx,  self.scenario_data.checkers_list, game_data_vector, self.environment_key) 
        reward /= 50  
        def update_reward(instruction_done_elem, reward_elem):
            return jax.lax.cond(instruction_done_elem, 
                                lambda _: reward_elem + 1, 
                                lambda _: reward_elem, 
                                operand=None)
            
        reward = update_reward(instruction_done, reward)
        done = instruction_done | done
        
        instruction_done_float = jnp.float32(instruction_done)
        new_episode_sr = env_state.success_rate + instruction_done_float
        
        state = TextEnvState(
            env_state=state, 
            timestep=state.timestep,
            instruction=instructions_emb,
            idx=idx,
            environment_key=environment_key,
            success_rate=new_episode_sr * (1 - done),  # исправлено "success_rate"
            total_success_rate=env_state.total_success_rate * (1 - done) + new_episode_sr * done  # исправлено "new_episode_sr"
        )
        info["SR"] = state.total_success_rate
        return obs, state, reward, done, info


if __name__=="__main__":
    instruction, checker = get_random_instruction_and_checker()
    print(instruction)