import jax.numpy as jnp
import jax.lax as lax
import jax
from flax.struct import dataclass
from functools import partial
from craftext.checkers.base_functions.state_adapter import GameData
from craftext.checkers_jax.target_state import AchievmentTargetState as ATS
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

@jax.jit
# @partial(jax.jit, static_argnames=("target_state"))
def at_time_block_placed(game_data: GameData, target_state: ATS) -> jax.Array:
    block_name = target_state.time_placement.block_type
    radius = target_state.time_placement.radius
    
    if game_data is None or game_data.states is None:
        return False

    game_map = game_data.states[0].map.game_map
    if game_map is None:
        return False

    player_position = game_data.states[0].variables.player_position
    if player_position is None:
        return False

    x, y = player_position
    region = safe_dynamic_slice(game_map, x, y, radius, 5)
    in_range = jnp.abs(game_data.states[0].variables.light_level - target_state.time_placement.time_state) <= 0.2
    return in_range & (region == target_state.time_placement.block_type).any()



