# instruction_wrapper.py

from dataclasses import dataclass
from typing import Any, Optional
import jax
import jax.numpy as jnp
from flax import linen as nn, struct
from gym import Wrapper
from gym import Env

from craftext.encoders.craftext_base_model_encoder import EncodeForm
from craftext.encoders.craftext_distilbert_model_encoder import DistilBertEncode

from craftext.instructions.scenarios.handlers.craftext_scenarious_no_lambda import ScenariosNoLambda
from craftext.adapters.state_adapter import GameData

from craftext.adapters.state_adapter_classic import GameDataClassic

from craftext.instructions.wrappers.utils import list_to_array
# from craftext.checkers_jax.time_constrained import at_time_block_placed
# from craftext.checkers_jax.building_star import is_cross_formed
from jax import tree_util

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
    
def get_checker_functions():
    from craftext.checkers_jax.achivments import conditional_achivments
    from craftext.checkers_jax.time_constrained import at_time_block_placed
    from craftext.checkers_jax.building_star import is_cross_formed
    from craftext.checkers_jax.building import is_line_formed
    from craftext.checkers_jax.building import is_square_formed
    from craftext.checkers_jax.conditional import conditional_placing
    from craftext.checkers_jax.relevant import place_object_relevant_to
    
    return [
        conditional_achivments,
        conditional_placing,
        place_object_relevant_to,
        is_line_formed,
        is_square_formed,
        is_cross_formed,
        at_time_block_placed
    ]

# @struct.dataclass
# class Scenarios(Enum):
#     CONDITIONAL_ACHIEVEMENTS = 0
#     CONDITIONAL_PLACING = 1
#     LOCALIZATION_PLACE = 2
    
#     BUILD_LINE = 3
#     BUILD_SQUARE = 4
#     BUILD_STAR = 5
    
#     TIME_CONSTRAINED_PLACEMENT = 6
class InstructionWrapperSeveralTasks(Wrapper):
    def __init__(self, env , config_name=None, scenario_handler_class=ScenariosNoLambda,
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
        self.scenario_arguments = (self.scenario_handler.scenario_data_jax.arguments)

        
        self.env = env
        self.steps = 0

        # Determine the environment key and state structure
        self.environment_key = self.scenario_handler.environment_key
        self.StateStructure = GameData if self.environment_key == 1 else GameDataClassic

        print("Initialized Instruction Wrapper with environment key:", self.environment_key)
        # print(self.StateStructure)
        self.n_instructions = len(self.scenario_handler.scenario_data.instructions_list)
        # print(self.scenario_handler.scenario_data.instructions_list)
        # print(len(self.scenario_handler.scenario_data.instructions_list))
        
        
        self.checkers = list(map(lambda x:  jax.vmap(x, in_axes=(None, 0)), get_checker_functions()))
        
        self.batched_scenario_args = tree_util.tree_map(
            lambda *xs: jnp.stack(xs),
            *self.scenario_arguments
        )
        #print(self.scenario_handler.scenario_data_jax.arguments)
        #exit()
    def scenario_switch(self, index, data, scenario_args):
        """
        index: an int (or array of ints if batched) telling switch which function to call.
        data: the input data (or one item from the batch).
        scenario_args: additional arguments (or one item from the batch).
        """
        return jax.lax.switch(
            index,
            self.checkers,   # tuple/list of candidate functions
            data,
            scenario_args
        )
    
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
        # self.checker_id =  self.scenario_handler.scenario_data_jax.scenario_checker[idx]
        # print(f'checker_id: {self.checker_id}')
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
        
        results = jax.lax.switch(
            env_state.checker_id,
            self.checkers,
            game_data_vector, self.batched_scenario_args
        )
        
        # results = jax.vmap(get_checker_functions()[self.scenario_handler.scenario_data_jax.scenario_checker[env_state.idx]], in_axes=(None, 0))(game_data_vector, self.batched_scenario_args)
        # print(results)
        # print(env_state.idx)
        # print(done)
        instructions_done = results[env_state.idx]
        # print(instructions_done)
        
        reward /= 50
        reward += (jnp.int32(instructions_done) & 2)
        done = instructions_done | done
   
        new_episode_sr = env_state.success_rate + jnp.float32(instructions_done)

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
        self.steps += 1
        return obs, state, reward, done, info
 
#TODO:
# враппер для проверки всех задачь разом.
# посмотреть trl GRPO trainer -> использовать transformers
# ONE -> вернуть.