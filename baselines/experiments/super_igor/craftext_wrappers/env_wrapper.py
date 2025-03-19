# instruction_wrapper.py

from dataclasses import dataclass
from typing import Any, Optional
import jax
import jax.numpy as jnp
from flax import linen as nn, struct
from gym import Wrapper

from jax import tree_map

from craftext.craftext_encoder import EncodeForm, DistilBertEncode
from craftext.craftext_scenarious_no_lambda import ScenariosNoLambda
from craftext.checkers.base_functions.state_adapter import GameData
from craftext.checkers.base_functions.state_adapter_craftax_classic import GameDataClassic
from craftext.checkers_jax.achivments import conditional_achivments

import jax.numpy as jnp
from typing import List, TypeVar, Type

T = TypeVar("T")

def list_to_array(lst: List[T]) -> T:
    """Convert a list of dataclass instances to a batched version with jnp.arrays."""
    if not lst:
        raise ValueError("Input list is empty.")

    cls: Type[T] = type(lst[0])  
    converted_data = {}

    for k, field in cls.__dataclass_fields__.items():
        values = [getattr(v, k) for v in lst]

        if isinstance(values[0], jnp.ndarray):
            converted_data[k] = jnp.stack(values, axis=0)  
        elif isinstance(values[0], (int, float, bool)):  
            converted_data[k] = jnp.array(values) 
        else:
            converted_data[k] = list_to_array(values) 
    return cls(**converted_data)

@struct.dataclass
class TextEnvState:
    env_state: Any
    timestep: int
    full_instruction: Optional[jax.Array]
    instruction: Optional[jax.Array]
    step_idx: int
    idx: int
    success_rate: float
    total_success_rate: float
    environment_key: int
    rng: int

from gymnax.environments import spaces, environment
from craftax.craftax_classic.envs.craftax_state import (
    EnvState,
    EnvParams,
)

def check_subvector(x, y):
    return jnp.all(x == y)

class CustomInstructionWrapper(Wrapper):
    def __init__(self, env, instruction):
        self.env = env
        self.castom_initial_instruction = jnp.array(self.env.scenario_handler.castom_initial_instruction(instruction)[0])
        print("CASTOM INSTRUCTION SHAPE: ", self.castom_initial_instruction.shape)
    
    def reset(self, _rng, env_params, instruction_idx=-1):
        """
        Resets the environment and selects a random instruction embedding or token for the new episode.
        """

        obs, state = self.env.reset(_rng, env_params)
        
        state = TextEnvState(
            env_state=state.env_state,
            timestep=state.timestep,
            full_instruction=self.castom_initial_instruction,
            instruction=self.castom_initial_instruction[0],
            step_idx=0,
            idx=state.idx,
            environment_key=state.environment_key,
            success_rate=state.success_rate,
            total_success_rate=0.0,
            rng=_rng
        )
         
        return obs, state
    
    def step(self, _rng, env_state, action, env_params):
         obs, state, reward, done, info = self.env.step(_rng, env_state, action, env_params)
         return obs, state, reward, done, info
        
        
class SIInstructionWrapper(Wrapper):
    def __init__(self, env, config_name=None, scenario_handler_class=ScenariosNoLambda,
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
        self.encoded_instruction = self.scenario_handler.scenario_data_jax.embeddings_list[0][0].reshape(1, -1)
        
        #print(self.encoded_instruction)
       # exit()
        self.scenario_arguments =list_to_array(self.scenario_handler.scenario_data_jax.arguments)
        self.env = env
        self.steps = 0
        self.end_embedding = self.scenario_handler.scenario_data_jax.embeddings_list[-1][-1]

        # Determine the environment key and state structure
        self.environment_key = self.scenario_handler.environment_key
        self.StateStructure = GameData if self.environment_key == 1 else GameDataClassic

        print("Initialized Instruction Wrapper with environment key:", self.environment_key)
        print(self.StateStructure)
        self.n_instructions = len(self.scenario_handler.scenario_data.instructions_list)
        print(self.scenario_handler.scenario_data.instructions_list)
        print(len(self.scenario_handler.scenario_data.instructions_list))
        
        #print(self.scenario_handler.scenario_data_jax.arguments)
        #exit()
    
    @property
    def num_actions(self) -> int:
        return 18

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(18)
    
    def reset(self, _rng, env_params, instruction_idx=-1):
        """
        Resets the environment and selects a random instruction embedding or token for the new episode.
        """

        obs, state = self.env.reset(_rng, env_params)
        
        idx = jax.lax.cond(
                instruction_idx == -1, 
                lambda: jax.random.randint(_rng, shape=(), minval=0, maxval=len(self.scenario_handler.scenario_data_jax.embeddings_list)),
                lambda: instruction_idx
            )
        instructions_emb = self.scenario_handler.scenario_data_jax.embeddings_list[idx]

        # Initialize the state with the selected instruction embedding/token and set success rates to zero
        state = TextEnvState(
            env_state=state,
            timestep=state.timestep,
            full_instruction=instructions_emb,
            instruction=instructions_emb[0],
            step_idx=0,
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
        step_idx = env_state.step_idx
        mask = jnp.where(action == 17, True, False)
        actions_plans = action
        new_step_idx =  jax.lax.cond(mask, lambda _: step_idx+1, lambda _: step_idx, operand=None)
        action = jax.lax.cond(mask, lambda _: 0, lambda _: action, operand=None)
        
        obs, state, reward, done, info = self.env.step(_rng, env_state.env_state, action, env_params)
        # Obtain the game data vector for the current state and check instruction completion
        game_data_vector = self.StateStructure.from_state(env_state.env_state, state, action)
        
        # Run all function over all game_data_vector (now only conditional_achivments) 
        conditional_achivments_vmap = jax.vmap(conditional_achivments, in_axes=(None, 0))
        results = conditional_achivments_vmap(game_data_vector, self.scenario_arguments)
        # Choose result releted instructions in current env
        instruction_done_on_step = results[env_state.idx]
        plans_ends = jnp.all(env_state.full_instruction[step_idx] == self.end_embedding)# jax.vmap(check_subvector, in_axes=(0, None))(env_state.full_instruction[step_idx], self.end_embedding)
        
        need_give_reward = instruction_done_on_step & plans_ends # Give reward only if instruction done and plan ends
        
        reward /= 50
        reward = jax.lax.cond(need_give_reward, lambda _: reward + 1, lambda _: reward, operand=None)
        done = plans_ends | done # End only with agent motivation or if it died
   
        new_episode_sr = env_state.success_rate + jnp.float32(need_give_reward)

        # Update state with the new success rates
        state = TextEnvState(
            env_state=state,
            timestep=state.timestep,
            full_instruction=env_state.full_instruction,
            instruction=env_state.full_instruction[step_idx],
            step_idx = new_step_idx, 
            idx=env_state.idx,
            environment_key=env_state.environment_key,
            success_rate=new_episode_sr * (1 - done),
            total_success_rate=env_state.total_success_rate * (1 - done) + new_episode_sr * done,
            rng=env_state.rng
        )
        
        # Update step information in info dictionary
        info.update({"SR": state.total_success_rate, "steps": self.steps})
        self.steps += 1
        return obs, state, reward, done, info
 