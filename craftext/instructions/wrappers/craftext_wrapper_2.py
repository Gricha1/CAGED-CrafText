# instruction_wrapper.py

from dataclasses import dataclass
from typing import Any, Optional
import jax
import jax.numpy as jnp
from flax import linen as nn, struct
from gym import Wrapper


from craftext.encoders.craftext_base_model_encoder import EncodeForm
from craftext.encoders.craftext_distilbert_model_encoder import DistilBertEncode

from craftext.instructions.scenarios.handlers.craftext_scenarious_no_lambda import ScenariosNoLambda
from craftext.adapters.state_adapter import GameData

from craftext.adapters.state_adapter_classic import GameDataClassic

from craftext.checkers_jax.achivments import conditional_achivments
from craftext.checkers_jax.time_constrained import at_time_block_placed
from craftext.checkers_jax.building_star import is_cross_formed
from craftext.checkers_jax.building import is_line_formed
from craftext.checkers_jax.building import is_square_formed
from craftext.checkers_jax.conditional import conditional_placing
from craftext.checkers_jax.relevant import place_object_relevant_to

from jax import tree_util
from collections import UserList

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
    checker_id: int
    
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

# @dataclass
# class checkers:


# @partial(jax.jit, static_argnums=(0,))
# def batch_check_scan(game_data, ts_list):
#     """
#     ts_list: python-список или pytree длины N из TargetState,
#              полностью статический (не трассируется).
#     game_data: единый объект для всех проверок.
#     """
#     def body_fn(idx, carry):
#         # carry — это массив Bool[N]
#         results = carry
#         ts = ts_list[idx]           # это чистый Python-объект
#         ok = generic_check(game_data, ts)  # ваша единая check-функция
#         return results.at[idx].set(ok)

#     N = len(ts_list)
#     init = jnp.zeros((N,), dtype=jnp.bool_)
#     return lax.fori_loop(0, N, body_fn, init)

from functools import partial
import jax
from jax import lax
import jax.numpy as jnp
from flax import struct
from craftext.checkers_jax.target_state import TargetState

# @partial(jax.jit, static_argnums=(0,))
def generic_check(game_data, target_state: TargetState, cheker_id: int) -> jnp.ndarray:
    

    def fa(ts: TargetState): return conditional_achivments(game_data,    ts.achievements)
    def fb(ts: TargetState): return conditional_placing(game_data,       ts.conditional_placing)
    def fc(ts: TargetState): return place_object_relevant_to(game_data,  ts.Localization_placing)
    def fd(ts: TargetState): return is_line_formed(game_data,           ts.building_line)
    def fe(ts: TargetState): return is_square_formed(game_data,         ts.building_square)
    def ff(ts: TargetState): return is_cross_formed(game_data,          ts.building_star)
    def fg(ts: TargetState): return at_time_block_placed(game_data,     ts.time_placement)

    fns = (fa, fb, fc, fd, fe, ff, fg)

    return lax.switch(cheker_id, fns, target_state)



# @partial(jax.jit, static_argnums=(0, 1,))
def batch_check_scan(game_data, ts_list, id, N):
    """
    ts_list: python-список или pytree длины N из TargetState,
             полностью статический (не трассируется).
    game_data: единый объект для всех проверок.
    """
    def body_fn(idx, carry):
        # carry — это массив Bool[N]
        results = carry
        ts = tree_util.tree_map(lambda arr: arr[idx], ts_list)          # это чистый Python-объект
        ok = generic_check(game_data, ts, id)  # ваша единая check-функция
        return results.at[idx].set(ok)

    init = jnp.zeros((N,), dtype=jnp.bool_)
    return lax.fori_loop(0, N, body_fn, init)



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
        self.scenario_arguments = self.scenario_handler.scenario_data_jax.arguments
        self.batched_scenario_args = tree_util.tree_map(
            lambda *xs: jnp.stack(xs),
            *self.scenario_arguments
        )
        self.env = env
        self.steps = 0

        # Determine the environment key and state structure
        self.environment_key = self.scenario_handler.environment_key
        self.StateStructure = GameData if self.environment_key == 1 else GameDataClassic

        print("Initialized Instruction Wrapper with environment key:", self.environment_key)
        print(self.StateStructure)
        self.n_instructions = len(self.scenario_handler.scenario_data.instructions_list)
        # print(self.scenario_handler.scenario_data.instructions_list)
        print(len(self.scenario_handler.scenario_data.instructions_list))
        
        print(len(self.scenario_handler.scenario_data_jax.arguments))
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
            rng=_rng,
            checker_id=self.scenario_handler.scenario_data_jax.scenario_checker[idx]
        )
        return obs, state

    def step(self, _rng, env_state, action, env_params):
        """
        Takes a step in the environment, checking if the instruction is done, updating success rate and rewards.
        """
        obs, state, reward, done, info = self.env.step(_rng, env_state.env_state, action, env_params)
        # Obtain the game data vector for the current state and check instruction completion
        game_data_vector = self.StateStructure.from_state(env_state.env_state, state, action)
        
        results = batch_check_scan(game_data_vector, self.batched_scenario_args, env_state.checker_id, self.n_instructions)
        # results = jax.lax.switch(env_state.checker_id, # сразу правильное проикидывать. 
        #                         (
        #                             jax.vmap(conditional_achivments,  in_axes=(None, 0)),
        #                             jax.vmap(conditional_placing,     in_axes=(None, 0)),
        #                             jax.vmap(place_object_relevant_to,in_axes=(None, 0)),
        #                             jax.vmap(is_line_formed,          in_axes=(None, 0)),
        #                             jax.vmap(is_square_formed,        in_axes=(None, 0)),
        #                             jax.vmap(is_cross_formed,         in_axes=(None, 0)),
        #                             jax.vmap(at_time_block_placed,    in_axes=(None, 0))
        #                         ),
        #                         game_data_vector, self.batched_scenario_args
        #     )
        # results = jax.vmap(is_square_formed, in_axes=(None, 1))(game_data_vector, self.batched_scenario_args)
        # проверить строительство все таки предметно
        # делаем конфиг где нет строительства, и только строительство
        #                                 )
        #         @struct.dataclass
        # class Scenarios:
        #     CONDITIONAL_ACHIEVEMENTS = 0
        #     CONDITIONAL_PLACING = 1
        #     LOCALIZATION_PLACE = 2
            
        #     BUILD_LINE = 3
        #     BUILD_SQUARE = 4
        #     BUILD_STAR = 5
            
        #     TIME_CONSTRAINED_PLACEMENT = 6
        # light_dinamic_batched = jnp.expand_dims(light_dinamic, env_state.num_envs)

        # print(f'game_data_vector.states[0].variables: {game_data_vector.states[0].variables}')
        # game_data_vector.states[0].variables.light_level_dinamic.set(light_dinamic)
        # game_data_vector.states[0].variables = game_data_vector.states[0].variables.replace(light_level_dinamic=light_dinamic)
        # Run all function over all game_data_vector (now only conditional_achivments) 
        # print(f'shape: {game_data_vector}')
        # print(f'shape:{self.scenario_arguments}')
        # print(f'len: {len(self.scenario_arguments)}')
        # at_time_block_placed_achivment = jax.vmap(at_time_block_placed, in_axes=(None, 0))
        # 
        # print(type(self.scenario_arguments))
        # print(self.scenario_arguments.shape)
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
            rng=env_state.rng,
            checker_id=env_state.checker_id
        )
        
        # Update step information in info dictionary
        info.update({"SR": state.total_success_rate, "steps": self.steps})
        info.update({"Cheker_id": env_state.checker_id})
        self.steps += 1
        return obs, state, reward, done, info
 
#TODO:
# враппер для проверки всех задачь разом.
# посмотреть trl GRPO trainer -> использовать transformers
# ONE -> вернуть.