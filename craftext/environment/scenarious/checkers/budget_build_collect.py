import jax

from jax import (
    numpy as jnp,
    lax
)
from flax.struct import dataclass
from typing import Tuple

from typing import Union
from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic

from craftext.environment.scenarious.checkers.target_state import BuildSquareState
from craftext.environment.scenarious.checkers.squeres import (
    check_square_2x2, 
    check_square_3x3, 
    check_square_4x4
)
from craftext.environment.scenarious.checkers.target_state_cmdp_build_budget_collect import BuildBudgetState, CMDPTargetState

def checker_budget_build_collect(game_data: Union[GameDataClassic, GameData],  target_state: BuildBudgetState) -> jax.Array:
    raise NotImplementedError("checker_budget_build_collect is not implemented yet")
    block_type  = target_state.block_type
    
    return is_on_block(game_data, block_type)

def is_on_block(gd: Union[GameDataClassic, GameData], block_type):
    x, y = gd.states[0].variables.player_position
    game_map = gd.states[0].map.game_map
    return jnp.array_equal(game_map[x][y], block_type)


