# instruction_wrapper.py

from dataclasses import dataclass
from typing import Any, Optional
import jax
import jax.numpy as jnp
from flax import linen as nn, struct
from gym import Wrapper

from jax import tree_map

from craftext.encoders.craftext_base_model_encoder import EncodeForm, DistilBertEncode
from craftext.craftext_scenarious_no_lambda import ScenariosNoLambda
from craftext.adapters.state_adapter import GameData
from craftext.adapters.state_adapter_classic import GameDataClassic
# from craftext.checkers_jax.time_constrained import at_time_block_placed
from craftext.checkers_jax.building_star import is_cross_formed

@struct.dataclass
class TextEnvState:
    env_state: Any
    timestep: int
    instruction: Optional[jax.Array]
    idx: int
    success_rate: float
    total_success_rate: float
    environment_key: int
    rng: int
    
import jax.numpy as jnp
from typing import List, TypeVar, Type
from enum import Enum
T = TypeVar("T")

def list_to_array(lst: List[T]) -> T:
    """Convert a list of dataclass instances to a batched version with jnp.arrays."""
    if not lst:
        raise ValueError("Input list is empty.")

    cls: Type[T] = type(lst[0])  # Определяем класс элементов списка
    converted_data = {}

    for k, field in cls.__dataclass_fields__.items():
        values = [getattr(v, k) for v in lst]

        # Если поле уже является jnp.ndarray, то стекуем его вдоль первой оси
        if isinstance(values[0], jnp.ndarray):
            converted_data[k] = jnp.stack(values, axis=0)  # Собираем массив массивов
        elif isinstance(values[0], (int, float, bool)):  
            converted_data[k] = jnp.array(values)  # Просто массив скаляров
        else:
            converted_data[k] = list_to_array(values)  # Рекурсивный вызов для вложенных датаклассов

    return cls(**converted_data)


class InstructionWrapper(Wrapper):
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
        self.encoded_instruction = self.scenario_handler.initial_instruction
        self.scenario_arguments = list_to_array(self.scenario_handler.scenario_data_jax.arguments)
        self.env = env
        self.steps = 0

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
        obs, state, reward, done, info = self.env.step(_rng, env_state.env_state, action, env_params)
        # Obtain the game data vector for the current state and check instruction completion
        game_data_vector = self.StateStructure.from_state(env_state.env_state, state, action)
        
        # light_dinamic_batched = jnp.expand_dims(light_dinamic, env_state.num_envs)

        # print(f'game_data_vector.states[0].variables: {game_data_vector.states[0].variables}')
        # game_data_vector.states[0].variables.light_level_dinamic.set(light_dinamic)
        # game_data_vector.states[0].variables = game_data_vector.states[0].variables.replace(light_level_dinamic=light_dinamic)
        # Run all function over all game_data_vector (now only conditional_achivments) 
        # print(f'shape: {game_data_vector}')
        # print(f'shape:{self.scenario_arguments}')
        # print(f'len: {len(self.scenario_arguments)}')
        # at_time_block_placed_achivment = jax.vmap(at_time_block_placed, in_axes=(None, 0))
        at_time_block_placed_achivment = jax.vmap(is_cross_formed, in_axes=(None, 0))

        # print(type(self.scenario_arguments))
        # print(self.scenario_arguments.shape)
        results = at_time_block_placed_achivment(game_data_vector, self.scenario_arguments)
        # Choose result releted instructions in current env
        instruction_done = results[env_state.idx]

        reward /= 50
        reward = jax.lax.cond(instruction_done, lambda _: reward + 1, lambda _: reward, operand=None)
        done = instruction_done | done
   
        new_episode_sr = env_state.success_rate + jnp.float32(instruction_done)

        # Update state with the new success rates
        state = TextEnvState(
            env_state=state,
            timestep=state.timestep,
            instruction=env_state.instruction,
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
 