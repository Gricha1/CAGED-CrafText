import jax.numpy as jnp
import jax.lax as lax
import jax
from flax.struct import dataclass

from craftext.adapters.state_adapter import GameData
from craftext.checkers_jax.target_state import TargetState, BuildStarState, BuildStarState
from functools import partial


import jax
import jax.numpy as jnp

def check_cross(region: jax.Array, stone_index: int, size: int, cross_type: int) -> jax.Array:
    center = size // 2

    def compute_straight(_):
        return (jnp.all(region[center, :] == stone_index) &
                jnp.all(region[:, center] == stone_index))

    def compute_diagonal(_):
        return (jnp.all(jnp.diag(region) == stone_index) &
                jnp.all(jnp.diag(jnp.fliplr(region)) == stone_index))


    def compute_size3(_):
        straight = compute_straight(None)
        diagonal = compute_diagonal(None)
        return straight | diagonal


    def compute_size5_7(_):
        straight = compute_straight(None)
        diagonal = compute_diagonal(None)
        return straight | diagonal | (straight & diagonal)

    base_result = jax.lax.cond(
        size == 3,
        compute_size3,
        lambda _: jax.lax.cond(
            (size == 5) | (size == 7),
            compute_size5_7,
            lambda _: jnp.array(False), 
            operand=None
        ),
        operand=None
    )


    final_result = jax.lax.cond(
        cross_type == 0,
        lambda _: compute_straight(None),
        lambda _: jax.lax.cond(
            cross_type == 1,
            lambda _: compute_diagonal(None),
            lambda _: base_result,
            operand=None
        ),
        operand=None
    )

    return final_result


@dataclass
class Carry:
    region: jax.Array
    stone_index: int
    region_size: int
    size: int
    cross_type: int

def scan_cross_function(carry: Carry, x):
    i, j = x // carry.region_size, x % carry.region_size
    
    
    condition = jnp.logical_or(i + carry.size > carry.region_size,
                            j + carry.size > carry.region_size)
    
    def branch_true(_):
        sub_region= carry.region[i:i+carry.size, j:j+carry.size]
        is_cross = check_cross(sub_region, carry.stone_index, carry.size, carry.cross_type)
        
        return is_cross

    return carry, lax.cond(condition, branch_true, lambda _: False, operand=None)

def is_cross_formed(game_data: GameData,  target_state: BuildStarState) -> jax.Array:
    
    block_index = target_state.block_type
    radius = 10
    size = target_state.size
    cross_type = target_state.cross_type

    return jax.lax.select(target_state.need_to_achieve, 
                   cross_checker(game_data, block_index, radius, size, cross_type),
                   jnp.array(False))
    
def cross_checker(game_data: GameData, block_index: int, radius: int, size: int, cross_type: int) -> jax.Array:

    game_map = game_data.states[0].map.game_map
    
    player_position = game_data.states[0].variables.player_position
    
    x, y = player_position
    region_size = 2 * radius + 1
    region = lax.dynamic_slice(
        game_map,
        start_indices=(x - radius, y - radius),
        slice_sizes=(region_size, region_size)
    )

    indixes = jnp.arange(region_size*region_size)
    carry = Carry(region=region,
            stone_index=block_index,
            region_size=region_size,
            size=size,
            cross_type=cross_type)
    _, crosses = lax.scan(scan_cross_function, carry, indixes)
    
    return jnp.any(crosses)

# # 89.169.171.236

