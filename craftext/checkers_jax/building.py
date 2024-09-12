import jax
import jax.numpy as jnp
from jax import lax
from typing import List

### CROSS

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

def check_cross(center, game_map, stone_index):
    """
    Checks if a cross pattern of stones is formed.
    """
    i, j = center
    return jnp.all(jnp.array([
        game_map[i, j] == stone_index,     
        game_map[i - 1, j] == stone_index, 
        game_map[i + 1, j] == stone_index, 
        game_map[i, j - 1] == stone_index, 
        game_map[i, j + 1] == stone_index, 

        game_map[i-1, j-1] != stone_index,
        game_map[i+1, j+1] != stone_index,
        game_map[i-1, j+1] != stone_index,
        game_map[i+1, j-1] != stone_index
    ]))

def scan_function(carry, x):
    """
    Scans the region to check for cross patterns.
    """
    game_map, stone_index, region_size = carry
    i, j = x // region_size, x % region_size
    is_cross = check_cross((i, j), game_map, stone_index)
    return carry, is_cross

def is_cross_formed(game_data, block_name: str, radius: int = 5) -> bool:
    stone_index = blocks_list.index(block_name)
    
    game_map = game_data.states[0].map.game_map[0]
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

    indices = jnp.arange(region_size * region_size)

    carry = (region, stone_index, region_size)
    _, crosses = lax.scan(scan_function, carry, indices)
    return jnp.any(crosses)

####  SQUERE

def check_square(center, region, stone_index):
    """
    Проверка наличия квадрата из блоков указанного размера.
    """
    i, j = center

    return jnp.all(jnp.array([
            # Проверка центрального квадрата 3x3 из блоков stone_index
            region[i, j] == stone_index,
            region[i-1, j] == stone_index,
            region[i+1, j] == stone_index,
            region[i, j-1] == stone_index,
            region[i, j+1] == stone_index,
            region[i-1, j-1] == stone_index,
            region[i-1, j+1] == stone_index,
            region[i+1, j-1] == stone_index,
            region[i+1, j+1] == stone_index,

            # Проверка внешней границы 5x5 без блока stone_index
            region[i-2, j-2] != stone_index,
            region[i-2, j-1] != stone_index,
            region[i-2, j] != stone_index,
            region[i-2, j+1] != stone_index,
            region[i-2, j+2] != stone_index,
            region[i-1, j-2] != stone_index,
            region[i, j-2] != stone_index,
            region[i+1, j-2] != stone_index,
            region[i+2, j-2] != stone_index,
            region[i+2, j-1] != stone_index,
            region[i+2, j] != stone_index,
            region[i+2, j+1] != stone_index,
            region[i+2, j+2] != stone_index,
            region[i-1, j+2] != stone_index,
            region[i, j+2] != stone_index,
            region[i+1, j+2] != stone_index
        ]))

    #return jnp.all(square_blocks == stone_index)
  #  return False

def scan_square_function(carry, x):
    region, stone_index, region_size, size = carry
    i, j = x // region_size, x % region_size
    is_square = check_square((i, j), region, stone_index)
    return carry, is_square


def is_square_formed(game_data, block_name: str, size: int = 2, radius: int = 5) -> bool:
    """
    Проверка на образование квадрата указанного размера из блоков в радиусе вокруг позиции игрока.
    """
    # Получаем индекс блока по имени
    stone_index = blocks_list.index(block_name)

    # Получаем карту
    game_map = game_data.states[0].map.game_map[0]
    if game_map is None:
        return False

    # Получаем позицию игрока
    player_position = game_data.states[0].variables.player_position
    if player_position is None:
        return False

    x, y = player_position

    # Определяем размеры области вокруг игрока
    region_size = 2 * radius + 1

    # Используем lax.dynamic_slice для извлечения области карты вокруг игрока
    region = lax.dynamic_slice(
        game_map,
        start_indices=(x - radius, y - radius),
        slice_sizes=(region_size, region_size)
    )

    # Создаем список индексов для обхода области
    indices = jnp.arange(region_size * region_size)

    # Передаем регион и индекс камня в качестве переносимых данных
    carry = (region, stone_index, region_size, size)
    
    # Используем lax.scan для проверки всех возможных квадратов в пределах области
    _, squares = lax.scan(scan_square_function, carry, indices)
    
    # Проверяем, найден ли хотя бы один квадрат
    return jnp.any(squares)