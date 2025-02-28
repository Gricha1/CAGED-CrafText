import jax.numpy as jnp
import jax.lax as lax
import jax
from flax.struct import dataclass
from enum import Enum

class PatternType:
    CROSS = jnp.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
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

def scale_pattern(pattern: jax.Array, size: int) -> jax.Array:
    """
    Масштабирует базовый паттерн до заданного размера, сохраняя его форму.
    """
    base_size = pattern.shape[0]
    if base_size == size:
        return pattern
    
    scaled_pattern = jnp.zeros((size, size), dtype=jnp.int32)
    center_offset = (size - base_size) // 2
    
    for i in range(base_size):
        for j in range(base_size):
            if pattern[i, j] == 1:
                scaled_i = int(i * size / base_size)
                scaled_j = int(j * size / base_size)
                scaled_pattern = scaled_pattern.at[scaled_i, scaled_j].set(1)
    
    return scaled_pattern

def get_pattern(pattern_type: PatternType, size: int) -> jax.Array:
    """
    Получает масштабированный шаблон.
    """
    return scale_pattern(pattern_type, size)

def check_pattern(region: jax.Array, stone_index: int, pattern: jax.Array) -> jax.Array:
    """
    Проверяет, соответствует ли подрегион переданному шаблону.
    """
    pattern_size = pattern.shape[0]
    sub_region = lax.dynamic_slice(region, (0, 0), (pattern_size, pattern_size))

    # Получаем индексы, где pattern == 1
    mask_indices = jnp.where(pattern == 1)
    
    # Используем jnp.take_along_axis для извлечения соответствующих значений
    selected_elements = jnp.take_along_axis(sub_region, mask_indices, axis=None)

    return jnp.all(selected_elements == stone_index)


@dataclass
class Carry:
    region: jax.Array
    stone_index: int
    region_size: int
    pattern: jax.Array

def scan_pattern_function(carry: Carry, x):
    i, j = x // carry.region_size, x % carry.region_size
    pattern_size = carry.pattern.shape[0]
    
    out_of_bounds = (i + pattern_size > carry.region_size) | (j + pattern_size > carry.region_size)
    
    def compute(_):
        sub_region = lax.dynamic_slice(carry.region, (i, j), (pattern_size, pattern_size))
        return check_pattern(sub_region, carry.stone_index, carry.pattern)
    
    result = lax.cond(out_of_bounds, lambda _: jnp.array(False), compute, operand=None)
    
    return carry, result

def is_pattern_formed(game_data, block_name, pattern_type: PatternType, size=3, radius=5):
    """
    Проверяет, сформирован ли заданный шаблон в окрестности игрока.
    """
    stone_index = block_name
    
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
    pattern = get_pattern(pattern_type, size)
    
    region = lax.dynamic_slice(
        game_map,
        start_indices=(x - radius, y - radius),
        slice_sizes=(region_size, region_size)
    )
    
    indices = jnp.arange(region_size * region_size)
    carry = Carry(region, stone_index, region_size, pattern)
    _, matches = lax.scan(scan_pattern_function, carry, indices)
    
    return jnp.any(matches).astype(bool)
