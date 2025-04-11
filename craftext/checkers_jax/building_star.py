import jax.numpy as jnp
import jax.lax as lax
import jax
from flax.struct import dataclass

from craftext.adapters.state_adapter import GameData
from craftext.checkers_jax.target_state import TargetState
from functools import partial

# Blocks list as an example
blocks_list = [
    "INVALID", "OUT_OF_BOUNDS", "GRASS", "WATER", "STONE", "TREE", 
    "WOOD", "PATH", "COAL", "IRON", "DIAMOND", "CRAFTING_TABLE", 
    "FURNACE", "SAND", "LAVA", "PLANT", "RIPE_PLANT", "WALL", 
    "DARKNESS", "WALL_MOSS", "STALAGMITE", "SAPPHIRE", "RUBY", 
    "CHEST", "FOUNTAIN", "FIRE_GRASS", "ICE_GRASS", "GRAVEL", 
    "FIRE_TREE", "ICE_SHRUB", "ENCHANTMENT_TABLE_FIRE", 
    "ENCHANTMENT_TABLE_ICE", "NECROMANCER", "GRAVE", "GRAVE2", 
    "GRAVE3", "NECROMANCER_VULNERABLE"
]

def safe_dynamic_slice(game_map, x, y, radius, max_radius):
    full_region_size = 2 * max_radius + 1
    padded_map = jnp.pad(game_map, max_radius, mode='constant')

    x_padded = x + max_radius
    y_padded = y + max_radius

    region = lax.dynamic_slice(
        padded_map,
        start_indices=(x_padded - max_radius, y_padded - max_radius),
        slice_sizes=(full_region_size, full_region_size)
    )

    # Затем обрезаем (маскируем) лишнее, так как radius может быть меньше max_radius
    coord_range = jnp.arange(full_region_size) - max_radius
    mask_x = jnp.abs(coord_range) <= radius
    mask_y = mask_x[:, None]
    mask = mask_x & mask_y

    region_masked = jnp.where(mask, region, 0)
    return region_masked

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
        straight = compute_straight()
        diagonal = compute_diagonal(None)
        return straight | diagonal


    def compute_size5_7(_):
        straight = compute_straight()
        diagonal = compute_diagonal()
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
        sub_region = safe_dynamic_slice(carry.region, i, j, carry.size, 10)
        is_cross = check_cross(sub_region, carry.stone_index, carry.size, carry.cross_type)
        
        return is_cross

    return carry, lax.cond(condition, branch_true, lambda _: False, operand=None)


def is_cross_formed(game_data: GameData, target_state: TargetState) -> jax.Array:


    block_name = target_state.building_star.block_type
    radius = target_state.building_star.radius
    size = target_state.building_star.size 
    cross_type = target_state.building_star.cross_type
    
    stone_index = block_name
    
    if game_data is None or game_data.states is None:
        return jnp.array(False)
    
    game_map = game_data.states[0].map.game_map
    if game_map is None:
        return jnp.array(False)
    
    player_position = game_data.states[0].variables.player_position
    if player_position is None:
        return jnp.array(False)
    
    x, y = player_position
    region_size = 2 * 10 + 1
    # print(f"region_size: {region_size}")
    region = safe_dynamic_slice(
        game_map,
        x,
        y,
        radius,
        10
    )
    indices = jnp.arange(0, region_size * region_size)
    carry = Carry(region=region,
            stone_index=stone_index,
            region_size=region_size,
            size=size,
            cross_type=cross_type)
    _, crosses = lax.scan(scan_cross_function, carry, indices)
    
    return jnp.any(crosses)

# 89.169.171.236

