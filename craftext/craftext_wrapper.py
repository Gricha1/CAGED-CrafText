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
from craftext.craftext_encoder import EncodeModel, EncodeForm, DistilBertEncode
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
    rng: int

class InstructionWrapper(Wrapper):
    def __init__(self, env, config_name=None, scenario_handler_class=CrafTextScenarios,
                  encode_model_class=DistilBertEncode, encode_form=EncodeForm.EMBEDDING):
        """
        Initializes the InstructionWrapper with the environment, creating EncodeModel and CrafTextScenarios.
        
        Parameters:
        - env: The environment to wrap.
        - config_name: Optional configuration name for scenarios.
        - encode_model_class: A class for the encoding model. Defaults to DistilBertEncode.
        - encode_form: The form of encoding (EMBEDDING or TOKEN). Defaults to EMBEDDING.
        """
        super().__init__(env)

        # Initialize the encoding model using the provided class
        self.encode_model = encode_model_class(form_to_use=encode_form)

        # Initialize the scenario handler with the encoding model
        self.scenario_handler = scenario_handler_class(self.encode_model, config_name)
        self.encoded_instruction = self.scenario_handler.initial_instruction
        self.env = env
        self.steps = 0

        # Determine the environment key and state structure
        self.environment_key = self.scenario_handler.environment_key
        self.StateStructure = GameData if self.environment_key == 1 else GameDataClassic

        print("Initialized Instruction Wrapper with environment key:", self.environment_key)
        print(self.StateStructure)
       # exit()
    
    def reset(self, _rng, env_params):
        """
        Resets the environment and selects a random instruction embedding or token for the new episode.
        """

        obs, state = self.env.reset(_rng, env_params)
        
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
            total_success_rate=0.0,
            rng=_rng
        )
        return obs, state

    def step(self, _rng, env_state, action, env_params):
        """
        Takes a step in the environment, checking if the instruction is done, updating success rate and rewards.
        """
        ### need to remove or rewrite
       # vector_rng = jnp.full_like(_rng, 42)
       # key = jax.random.PRNGKey(0)

        obs, state, reward, done, info = self.env.step(_rng, env_state.env_state, action, env_params)
        
        # Obtain the game data vector for the current state and check instruction completion
        game_data_vector = self.StateStructure.from_state(env_state.env_state, state, action)
        instruction_done = jax.lax.switch(env_state.idx, self.scenario_handler.scenario_data.checkers_list, game_data_vector, self.environment_key)

        # Normalize reward and increment if instruction is done
        reward /= 50
        reward = jax.lax.cond(instruction_done, lambda _: reward + 1, lambda _: reward, operand=None)
        done = instruction_done | done
        done_mask = jnp.array(done, dtype=jnp.bool_)
        # Update success rate and total success rate
        new_episode_sr = env_state.success_rate + jnp.float32(instruction_done)
        # Split the RNG only if done is True
        __rng, new_rng = jax.random.split(env_state.rng)
        new_rng = jnp.where(done_mask, new_rng, env_state.rng)  # Keep the current RNG if done is False

        # Generate idx only if done is True
        idx = jax.random.randint(new_rng, shape=(), minval=0, maxval=len(self.scenario_handler.scenario_data_jax.embeddings_list))
        idx = jnp.where(done_mask, idx, env_state.idx)  # Keep the current idx if done is False

        # Update state with the new success rates
        state = TextEnvState(
            env_state=state,
            timestep=state.timestep,
            instruction=env_state.instruction,
            idx=idx,
            environment_key=env_state.environment_key,
            success_rate=new_episode_sr * (1 - done),
            total_success_rate=env_state.total_success_rate * (1 - done) + new_episode_sr * done,
            rng=new_rng
        )
        
        # Update step information in info dictionary
        info.update({"SR": state.total_success_rate, "steps": self.steps})
        self.steps += 1
        return obs, state, reward, done, info
