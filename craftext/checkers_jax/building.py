import jax
from jax import (
    numpy as jnp,
    lax
)
from flax import struct
from typing import Callable, NamedTuple
from craftext.checkers_jax.squeres import (
    check_square_2x2, 
    check_square_3x3, 
    check_square_4x4
)
from craftext.adapters.state_adapter import GameData
from typing import Tuple 
from craftext.scenarios.constants import BlockType
from craftext.checkers_jax.target_state import TargetState, BuildSquareState, BuildLineState
from functools import partial
from jax import lax

@struct.dataclass
class Building:
    game_map: jax.Array
    stone_index: int
    region_size: int


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


def is_square_formed(game_data: GameData,  target_state: BuildSquareState) -> jax.Array:
    
    block_index = target_state.block_type
    radius = target_state.radius
    size = target_state.size
    
    return jax.lax.select(target_state.need_to_achieve, 
                   checker_square(game_data, block_index, radius, size),
                   jnp.array(False))
    
def checker_square(game_data: GameData, block_index: int, radius: int, size: int) -> jax.Array:
    """
    Проверка на образование квадрата указанного размера из блоков в радиусе вокруг позиции игрока.
    """

    game_map = game_data.states[0].map.game_map
    
    # Получаем карту
    
    # Получаем позицию игрока
    player_position = game_data.states[0].variables.player_position
    
    x, y = player_position

    # Определяем размеры области вокруг игрока
    region_size = 2 * 10 + 1

    # Используем lax.dynamic_slice для извлечения области карты вокруг игрока
    region = lax.dynamic_slice(
        game_map,
        start_indices=(x - radius, y - radius),
        slice_sizes=(region_size, region_size)
    )

    # Создаем список индексов для обхода области
    indices = jnp.arange(region_size * region_size)

    # Передаем регион и индекс камня в качестве переносимых данных
    carry = (region, block_index, region_size, size)
    
    # Используем lax.scan для проверки всех возможных квадратов в пределах области
    _, squares = lax.scan(scan_square_function, carry, indices)
    
    # Проверяем, найден ли хотя бы один квадрат
    return jnp.any(squares)

    



### LINE
from craftext.checkers_jax.lines import check_line_2, check_line_3, check_line_4

def check_line_by_size(center: Tuple[int, int], region: jax.Array, stone_index: int, size: int, check_diagonal: bool) -> Callable:
    i, j = center

    func = jax.lax.switch(
        size - 2, 
        [
            lambda: check_line_2((i, j), region, check_diagonal),
            lambda: check_line_3((i, j), region, check_diagonal),
            lambda: check_line_4((i, j), region, check_diagonal)
        ]
    )
        
    return func

def scan_line_function(carry, x):
    region, stone_index, region_size, size, check_diagonal = carry
    i, j = x // region_size, x % region_size
    is_line = check_line_by_size((i, j), region, stone_index, size, check_diagonal)
    return carry, is_line

def is_line_formed(game_data: GameData,  target_state: BuildLineState) -> jax.Array:
    
    block_index = target_state.block_type
    radius = target_state.radius
    size = target_state.size
    check_diagonal = target_state.is_diagonal

    return jax.lax.select(target_state.need_to_achieve, 
                   checker_line(game_data, block_index, radius, size, check_diagonal),
                   jnp.array(False))
    
def checker_line(game_data: GameData, block_index: int, radius: int, length: int, check_diagonal: bool) -> jax.Array:
    """
    Проверка на образование квадрата указанного размера из блоков в радиусе вокруг позиции игрока.
    """
    
    # Получаем карту
    game_map =  game_data.states[0].map.game_map
    
    binary_map = (game_map == block_index).astype(jnp.int32)
    
    # Получаем позицию игрока
    player_position = game_data.states[0].variables.player_position
    
    x, y = player_position


    # Определяем размеры области вокруг игрока
    region_size = 2 * 10 + 1

    # Используем lax.dynamic_slice для извлечения области карты вокруг игрока
    region = lax.dynamic_slice(
        binary_map,
        start_indices=(x - radius, y - radius),
        slice_sizes=(region_size, region_size)
    )

    # Создаем список индексов для обхода области
    indices = jnp.arange(region_size * region_size)

    # Передаем регион и индекс камня в качестве переносимых данных
    carry = (region, block_index, region_size, length, check_diagonal)
    
    # Используем lax.scan для проверки всех возможных квадратов в пределах области
    _, squares = lax.scan(scan_line_function, carry, indices)
    #print(squares)
    # Проверяем, найден ли хотя бы один квадрат
    return jnp.any(squares)


# def is_star_formed(game_data)