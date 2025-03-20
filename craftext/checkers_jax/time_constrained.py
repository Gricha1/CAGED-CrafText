import jax.numpy as jnp
import jax.lax as lax
import jax
from flax.struct import dataclass

from craftext.checkers.base_functions.state_adapter import GameData
from craftext.checkers_jax.target_state import TargetState
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

def at_time_block_placed(game_data: GameData, target_state: TargetState) -> jax.Array:
    block_name = target_state.building_star.block_type
    radius = target_state.building_star.radius
    
    if game_data is None or game_data.states is None:
        return False

    game_map = game_data.states[0].map.game_map
    if game_map is None:
        return False

    player_position = game_data.states[0].variables.player_position
    if player_position is None:
        return False

    x, y = player_position
    region_size = 2 * radius + 1

    region = lax.dynamic_slice(
        game_map,
        start_indices=(x - radius, y - radius),
        slice_sizes=(region_size, region_size)
    )
    
    return (region == block_name.value).any()