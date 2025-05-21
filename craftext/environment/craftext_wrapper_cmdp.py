from typing import Any, Optional
import jax
import jax.numpy as jnp
from jax import lax

from flax import struct
from gym import Wrapper


from craftext.environment.encoders.craftext_base_model_encoder import EncodeForm
from craftext.environment.encoders.craftext_distilbert_model_encoder import DistilBertEncode

from craftext.environment.scenarious.manager_cmdp import ScenariosNoLambdaCMDP

from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic
from craftext.environment.craftext_constants import Scenarios 
from craftext.environment.scenarious.checkers.achivments       import checker_acvievments
from craftext.environment.scenarious.checkers.time_constrained import checker_time_placement
from craftext.environment.scenarious.checkers.building_star    import checker_star
from craftext.environment.scenarious.checkers.building_line    import checker_line
from craftext.environment.scenarious.checkers.building_square  import checker_square
from craftext.environment.scenarious.checkers.conditional      import checker_conditional_placement
from craftext.environment.scenarious.checkers.relevant         import cheker_localization
from craftext.environment.scenarious.checkers.step_on_block    import checker_step_on_block
from typing import Union

from craftext.environment.craftext_wrapper import InstructionWrapper

@struct.dataclass
class TextEnvStateCMDP:
    env_state: Any
    timestep: int
    instruction: Optional[jax.Array]
    textual_constraint: Optional[jax.Array]
    idx: int
    success_rate: float
    total_success_rate: float
    environment_key: int
    rng: int
    instruction_done: bool
    checker_id: int
    

class CMDPInstructionWrapper(InstructionWrapper):
    def __init__(self, env, config_name=None, scenario_handler_class=ScenariosNoLambdaCMDP,
                  encode_model_class=DistilBertEncode, encode_form=EncodeForm.EMBEDDING):
        super().__init__(env, config_name=config_name, scenario_handler_class=scenario_handler_class,
                  encode_model_class=encode_model_class, encode_form=encode_form)

        self.encoded_textual_constraint = self.scenario_handler.scenario_data_jax.constraints_embeddings_list[0]

    def reset(self, _rng, env_params, instruction_idx=-1):
        obs, state = super().reset(_rng, env_params, instruction_idx=instruction_idx)

        textual_constraint_emb = self.scenario_handler.scenario_data_jax.constraints_embeddings_list[state.idx]
        state = TextEnvStateCMDP(
            env_state=state.env_state,
            timestep=state.timestep,
            instruction=state.instruction,
            textual_constraint=textual_constraint_emb,
            idx=state.idx,
            environment_key=state.environment_key,
            success_rate=state.success_rate,
            total_success_rate=state.total_success_rate,
            rng=state.rng,
            instruction_done=state.checker_id,
            checker_id=state.checker_id
        )
        return obs, state

    def step(self, _rng, env_state, action, env_params):
        obs, state, reward, done, info = super().step(_rng, env_state, action, env_params)

        game_data_vector = self.StateStructure.from_state(env_state.env_state, state.env_state, action)
        ts = self.batched_ts.select(env_state.idx)
        info["cost"] = checker_step_on_block(game_data_vector, ts.step_on_block).astype(float)
        if "episode_cost" not in info:
            info["episode_cost"] = info["cost"]
        else:
            info["episode_cost"] += info["cost"]

        state = TextEnvStateCMDP(
            env_state=state.env_state,
            timestep=state.timestep,
            instruction=state.instruction,
            textual_constraint=env_state.textual_constraint,
            idx=state.idx,
            environment_key=state.environment_key,
            success_rate=state.success_rate,
            total_success_rate=state.total_success_rate,
            rng=state.rng,
            instruction_done=state.checker_id,
            checker_id=state.checker_id
        )

        return obs, state, reward, done, info