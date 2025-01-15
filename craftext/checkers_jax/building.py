import jax
from jax import (
    numpy as jnp,
    lax
)
from craftext.checkers_jax.squeres import (
    check_square_2x2, 
    check_square_3x3, 
    check_square_4x4
)
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

def check_square_by_size(center, region, stone_index, size):
    i, j = center

    return jax.lax.switch(
        size - 2, 
        [
            lambda: check_square_2x2((i, j), region, stone_index),
            lambda: check_square_3x3((i, j), region, stone_index),
            lambda: check_square_4x4((i, j), region, stone_index)
        ]
    )

def scan_square_function(carry, x):
    region, stone_index, region_size, size = carry
    i, j = x // region_size, x % region_size
    is_square = check_square_by_size((i, j), region, stone_index, size)
    return carry, is_square


def is_square_formed(game_data,  ix:int, block_name: str, size: int = 2, radius: int = 5) -> bool:
    """
    Проверка на образование квадрата указанного размера из блоков в радиусе вокруг позиции игрока.
    """
    # Получаем индекс блока по имени
    stone_index = block_name.value

    # Получаем карту
    game_map = game_data.states[0].map.game_map
    
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


### LINE
from craftext.checkers_jax.lines import check_line_2, check_line_3, check_line_4

def check_line_by_size(center, region, stone_index, size, check_diagonal):
    i, j = center

    return jax.lax.switch(
        size - 2, 
        [
            lambda: check_line_2((i, j), region, check_diagonal),
            lambda: check_line_3((i, j), region, check_diagonal),
            lambda: check_line_4((i, j), region, check_diagonal)
        ]
    )

def scan_line_function(carry, x):
    region, stone_index, region_size, size, check_diagonal = carry
    i, j = x // region_size, x % region_size
    is_line = check_line_by_size((i, j), region, stone_index, size, check_diagonal)
    return carry, is_line



def is_line_formed(game_data, ix:int, block_name: int, size: int = 2, check_diagonal:bool = False) -> bool:
    """
    Проверка на образование квадрата указанного размера из блоков в радиусе вокруг позиции игрока.
    """
    radius = 5
    stone_index = block_name.value
    # Получаем карту
    game_map =  game_data.states[0].map.game_map
    
    binary_map = (game_map == stone_index).astype(jnp.int32)
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
        binary_map,
        start_indices=(x - radius, y - radius),
        slice_sizes=(region_size, region_size)
    )

    # Создаем список индексов для обхода области
    indices = jnp.arange(region_size * region_size)

    # Передаем регион и индекс камня в качестве переносимых данных
    carry = (region, stone_index, region_size, size, check_diagonal)
    
    # Используем lax.scan для проверки всех возможных квадратов в пределах области
    _, squares = lax.scan(scan_line_function, carry, indices)
    #print(squares)
    # Проверяем, найден ли хотя бы один квадрат
    return jnp.any(squares)
