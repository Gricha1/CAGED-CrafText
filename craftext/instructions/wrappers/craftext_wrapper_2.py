# instruction_wrapper.py

from typing import Any, Optional
import jax
import jax.numpy as jnp
from jax import lax

from flax import struct
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

from craftext.checkers_jax.target_state import TargetState

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
    

# @partial(jax.jit, static_argnums=(0,))
def generic_check(game_data, target_state: TargetState, cheker_id: int) -> jnp.ndarray:
    

    def ca(ts: TargetState):  return conditional_achivments(game_data,    ts.achievements)
    def cp(ts: TargetState):  return conditional_placing(game_data,       ts.conditional_placing)
    def port(ts: TargetState):return place_object_relevant_to(game_data,  ts.Localization_placing)
    def ilf(ts: TargetState): return is_line_formed(game_data,            ts.building_line)
    def isf(ts: TargetState): return is_square_formed(game_data,          ts.building_square)
    def icf(ts: TargetState): return is_cross_formed(game_data,           ts.building_star)
    def atp(ts: TargetState): return at_time_block_placed(game_data,      ts.time_placement)

    fns = (ca, cp, port, ilf, isf, icf, atp)

    return lax.switch(cheker_id, fns, target_state)


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

        self.encode_model = encode_model_class(form_to_use=encode_form)

        # Initialize the scenario handler with the encoding model
        self.scenario_handler = scenario_handler_class(self.encode_model, config_name)
        self.encoded_instruction = self.scenario_handler.initial_instruction
        self.scenario_arguments = self.scenario_handler.scenario_data_jax.arguments
        self.batched_ts: TargetState = TargetState.stack(self.scenario_arguments)

        self.env = env
        self.steps = 0

        # Determine the environment key and state structure
        self.environment_key = self.scenario_handler.environment_key
        self.StateStructure = GameData if self.environment_key == 1 else GameDataClassic

        print("Initialized Instruction Wrapper with environment key:", self.environment_key)
        # print(self.StateStructure)
        self.n_instructions = len(self.scenario_handler.scenario_data.instructions_list)


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
                    
        ts = self.batched_ts.select(env_state.idx)
        results = generic_check(game_data_vector, ts, env_state.checker_id)
        instruction_done = results

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