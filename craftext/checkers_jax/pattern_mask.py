import jax.numpy as jnp
import jax.lax as lax
import jax
from flax.struct import dataclass
from enum import Enum

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

class PatternType:
    CROSS = jnp.array([
        [1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1],
    ])
    SQUARE = jnp.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])
    LINE = jnp.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]
    ])
    DIAGONAL = jnp.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    STAR = jnp.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ])

def transform_pattern(pattern: jax.Array, stone_index, size: int) -> jax.Array:

    #only pattern 
    return pattern * stone_index


def get_pattern(pattern_type: jax.Array, stone_index, size: int) -> jax.Array:
    return transform_pattern(pattern_type, stone_index, size)


def check_pattern(sub_region: jax.Array, pattern: jax.Array) -> jax.Array:
    mask_indices = pattern > 0
    return (mask_indices == sub_region).all()


@dataclass
class Carry:
    region: jax.Array
    stone_index: int
    region_size: int
    pattern: jax.Array

def scan_pattern_function(carry: Carry, x: int):
    position = x // carry.region_size, x % carry.region_size

    sub_region = lax.dynamic_slice(carry.region, position, carry.pattern.shape)
    
    if sub_region.shape[0] < carry.pattern.shape[0] or sub_region.shape[1] < carry.pattern.shape[1]:
        return carry, jnp.array(False)
     
    return carry, check_pattern(sub_region, carry.pattern)

def is_pattern_formed(game_data, block_name: str, pattern_type: jax.Array, size=3, radius=5):

    stone_index = blocks_list.index(block_name)

    if game_data is None or game_data.states is None:
        return False

    game_map = game_data.states[0].map.game_map
    if game_map is None:
        return False

    player_position = game_data.states[0].variables.player_position
    if player_position is None:
        return False

    x, y = player_position
    pattern = get_pattern(pattern_type, stone_index, size)
    region_size = 2 * radius + 1

    region = lax.dynamic_slice(
        game_map,
        start_indices=(x - radius, y - radius),
        slice_sizes=(region_size, region_size)
    )
    print(region)
    indices = jnp.arange(region_size * region_size)
    print(indices)
    carry = Carry(region, stone_index, region_size, pattern)
    _, matches = lax.scan(scan_pattern_function, carry, indices)
    return matches.any()
