# instruction_wrapper.py

import os
import random
import importlib
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import yaml
import jax
import jax.numpy as jnp
from jax import lax, vmap
import numpy as np
from flax import linen as nn, struct
from gym import Wrapper
from transformers import AutoModel, AutoTokenizer
import distrax

# Importing custom classes from your modules
from craftext.craftext_encoder import EncodeModel, EncodeForm
from craftext.craftext_scenarious import CrafTextScenarios, ScenarioDataJAX
from craftext.checkers.base_functions.state_adapter import GameData
from craftext.checkers.base_functions.state_adapter_craftax_classic import GameDataClassic
# from craftext.scenarios_loader import (
#     load_scenarios,
#     parse_craftext_settings,
#     load_config_or_env,
#     get_configs_path,
# )

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
    def __init__(self, env, config_name=None):
        """
        Initializes the InstructionWrapper with the environment, creating EncodeModel and CrafTextScenarios.
        """
        super().__init__(env)

        model_name="distilbert-base-uncased"
        encode_form=EncodeForm.EMBEDDING

        self.encode_model = EncodeModel(model_name=model_name, form_to_use=encode_form)
        self.scenario_handler = CrafTextScenarios(self.encode_model, config_name)
        self.encoded_instruction = self.scenario_handler.initial_instruction
        self.env = env
        self.steps = 0
        self.environment_key = self.scenario_handler.environment_key
        self.StateStructure = GameData if self.environment_key == 1 else GameDataClassic
        print("Initialized Instruction Wrapper with environment key:", self.environment_key)
    
    def reset(self, _rng, env_params):
        """
        Resets the environment and selects a random instruction embedding or token for the new episode.
        """
        obs, state = self.env.reset(_rng, env_params)
        
        # Select a random index for an instruction
        idx = jax.random.randint(_rng, shape=(), minval=0, maxval=len(self.scenario_handler.scenario_data_jax.embeddings_list))
        instructions_emb = self.scenario_handler.scenario_data_jax.embeddings_list[idx]

        # Initialize the state with the selected instruction embedding/token and set success rates to zero
        state = TextEnvState(
            env_state=state,
            timestep=state.timestep,
            instruction=instructions_emb,
            idx=idx,
            environment_key=self.environment_key,
            success_rate=0.0,
            total_success_rate=0.0
        )
        return obs, state

    def step(self, _rng, env_state, action, env_params):
        """
        Takes a step in the environment, checking if the instruction is done, updating success rate and rewards.
        """
        obs, state, reward, done, info = self.env.step(_rng, env_state.env_state, action, env_params)
        
        # Obtain the game data vector for the current state and check instruction completion
        game_data_vector = self.StateStructure.from_state(env_state.env_state, state, action)
        instruction_done = jax.lax.switch(env_state.idx, self.scenario_handler.scenario_data.checkers_list, game_data_vector, self.environment_key)

        # Normalize reward and increment if instruction is done
        reward /= 50
        reward = jax.lax.cond(instruction_done, lambda _: reward + 1, lambda _: reward, operand=None)
        done = instruction_done | done

        # Update success rate and total success rate
        new_episode_sr = env_state.success_rate + jnp.float32(instruction_done)
        
        # Update state with the new success rates
        state = TextEnvState(
            env_state=state,
            timestep=state.timestep,
            instruction=env_state.instruction,
            idx=env_state.idx,
            environment_key=env_state.environment_key,
            success_rate=new_episode_sr * (1 - done),
            total_success_rate=env_state.total_success_rate * (1 - done) + new_episode_sr * done
        )
        
        # Update step information in info dictionary
        info.update({"SR": state.total_success_rate, "steps": self.steps})
        self.steps += 1
        return obs, state, reward, done, info
