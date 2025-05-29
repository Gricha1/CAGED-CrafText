import jax

from jax import (
    numpy as jnp,
    lax
)
from jax import tree_util

from typing import Union
from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic

from craftext.environment.scenarious.checkers.target_state_light_level import LightLevelState
from craftext.environment.craftext_constants import BlockType

from craftax.craftax.constants import Action as ActionExtend
from craftax.craftax_classic.constants import Action as ActionClassic
from typing import Union

# class Action(ActionExtend, ActionClassic):
#     pass

def checker_budget_drink_level(game_data: Union[GameDataClassic, GameData],  target_state: LightLevelState) -> jax.Array:
    # raise NotImplementedError("checker_budget_build_collect is not implemented yet")
    light_level = target_state.level
    return is_sleep_at_night(game_data, light_level)

def is_sleep_at_night(game_data: Union[GameDataClassic, GameData], light_level):
    is_sleep = game_data.states[0].variables.is_sleeping               # jax.Array с целочисленным кодом действия
    light_level = game_data.states[0].variables.light_level  # jax.Array с уровнем освещённости

    # Порог из целевого состояния:
    threshold = light_level                     # число или jax.Array


    # Проверяем условия:
    is_dark_enough = light_level <= threshold

    return jnp.logical_and(is_dark_enough, is_sleep)
    

