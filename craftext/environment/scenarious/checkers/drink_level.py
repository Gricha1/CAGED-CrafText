import jax

from jax import (
    numpy as jnp,
    lax
)
from jax import tree_util

from typing import Union
from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic

from craftext.environment.scenarious.checkers.target_state_cmdp_budget_drink_level import DrinkLevelState
from craftext.environment.craftext_constants import BlockType

def checker_budget_drink_level(game_data: Union[GameDataClassic, GameData],  target_state: DrinkLevelState) -> jax.Array:
    # raise NotImplementedError("checker_budget_build_collect is not implemented yet")
    level = target_state.level
    return level_status(game_data, level)

def level_status(game_data: Union[GameDataClassic, GameData], level: int):
    current_level = game_data.states[0].variables.player_thirst

    # def get_item(index, inventory):
    #     leaves, _ = tree_util.tree_flatten(inventory)
    #     leaves = jnp.stack(leaves)
    #     return leaves[index]
    
    
    
    
    # curr = get_item(block_type, game_data.states[0].inventory)
    # prev = get_item(block_type, game_data.states[1].inventory)
    return current_level >= level


