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
from craftext.environment.scenarious.checkers.budget_build_collect import checker_budget_build_collect
from craftext.environment.scenarious.checkers.dont_move import checker_moveing_at_night_level
from craftext.environment.scenarious.checkers.sleep_at_night import checker_sleep_at_night
from craftext.environment.scenarious.checkers.monster_is_attacked_without_sword import checker_monster_is_attacked_without_sword
from craftext.environment.scenarious.checkers.drink_level import checker_budget_drink_level
from craftext.environment.scenarious.checkers.hp_level import checker_budget_hp_level
from craftext.environment.scenarious.checkers.hungry_level import checker_budget_hungry_level
from craftext.environment.scenarious.checkers.energy_level import checker_budget_energy_level

@struct.dataclass
class TextEnvStateCMDP:
    env_state: Any
    timestep: int
    instruction: Optional[jax.Array]
    textual_constraint: Optional[jax.Array]
    idx: int
    success_rate: float
    episode_cost: float
    cost: float
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

        self.config_name = config_name
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
            episode_cost=0.,
            cost=0.,
            total_success_rate=state.total_success_rate,
            rng=state.rng,
            instruction_done=state.checker_id,
            checker_id=state.checker_id
        )
        return obs, state

    def step(self, _rng, env_state, action, env_params):
        obs, state, reward, done, info = super().step(_rng, env_state, action, env_params)

        # set cost
        game_data_vector = self.StateStructure.from_state(env_state.env_state, state.env_state, action)
        ts = self.batched_ts.select(env_state.idx)

        if self.config_name == "simple_achivments_safe":
            cost = checker_step_on_block(game_data_vector, ts.step_on_block).astype(float)
        # budgetary
        elif self.config_name == "achievements_safe_budget_dont_move_night":
            cost = checker_moveing_at_night_level(game_data_vector, ts.night_constraint_level).astype(float)
        elif self.config_name == "build_squere_simple_safe_budget":
            cost = checker_budget_build_collect(game_data_vector, ts.build_budget_state).astype(float)
        elif self.config_name == "achievements_safe_budget_drink":
            cost = checker_budget_drink_level(game_data_vector, ts.drink_level_state).astype(float)
        elif self.config_name == "achievements_easy_safe_budget_drink":
            cost = checker_budget_drink_level(game_data_vector, ts.drink_level_state).astype(float)
        elif self.config_name == "achievements_safe_budget_energy":
            cost = checker_budget_energy_level(game_data_vector, ts.energy_level_state).astype(float)
        elif self.config_name == "achievements_safe_budget_hp":
            cost = checker_budget_hp_level(game_data_vector, ts.hp_level_state).astype(float)
        elif self.config_name == "achievements_safe_budget_hungry" or \
             self.config_name == "achievements_safe_budget_hungry_multi_limit" or \
             self.config_name == "cmdp_hard_achievements_budget_hungry":
            cost = checker_budget_hungry_level(game_data_vector, ts.hungry_level_state).astype(float)
        elif self.config_name == "achievements_safe_budget_sleep_at_night":
            cost = checker_sleep_at_night(game_data_vector, ts.night_constraint_level, ts.day_constraint_level).astype(float)
        # sequential
        elif self.config_name == "achievements_safe_sequential_defeat_monster":
            #assert 1 == 0
            cost = checker_monster_is_attacked_without_sword(game_data_vector).astype(float)
        else:
            assert 1 == 0, f"unknow config name: {self.config_name}, need assign cost function for this config"
            

        state = TextEnvStateCMDP(
            env_state=state.env_state,
            timestep=state.timestep,
            instruction=state.instruction,
            textual_constraint=env_state.textual_constraint,
            idx=state.idx,
            environment_key=state.environment_key,
            success_rate=state.success_rate,
            episode_cost=env_state.episode_cost + cost,
            cost=cost,
            total_success_rate=state.total_success_rate,
            rng=state.rng,
            instruction_done=state.checker_id,
            checker_id=state.checker_id
        )

        return obs, state, reward, done, info