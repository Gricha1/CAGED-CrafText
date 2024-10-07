import jax
from jax import jit
import jax.numpy as jnp
from jax import lax
from typing import List
from functools import partial

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
import jax
import jax.numpy as jnp
from jax import lax

def place_object_relevant_to(game_data, object_name, target_object_name, side, distance):
    """
    Check if the object is placed at a specific side (right, left, top, or bottom) and distance from the target_object_name
    within the area around the player (with a fixed radius of 15).

    Args:
    - game_data (GameData): The game data object containing the player's positions and map information.
    - object_name (str): The name of the block to check. Possible items: "STONE", "CRAFTING_TABLE", "FURNACE", "CHEST", "FOUNTAIN", "ENCHANTMENT_TABLE_FIRE", "ENCHANTMENT_TABLE_ICE", "PLANT".
    - target_object_name (str): The name of the target object.
    - side (int): The side on which the object should be placed. Encoded as:
                  0 = Right, 1 = Left, 2 = Top, 3 = Bottom.
    - distance (int): The distance at which the object should be placed from the target object.

    Returns:
    - bool: True if the object is placed at the specified side and distance relative to the target object, otherwise False.
    """
    # for original Craftax we need to define the map we want to use during traning
    # so, for Craftax-Classic we need to use just game_map
    # for Craftax game_map[0] where 0 is the first level of map
    game_map = game_data.states[0].map.game_map 
    player_position = game_data.states[0].variables.player_position

    if player_position is None:
        return False

    x, y = player_position

    # Фиксированный радиус
    radius = 5

    # Определяем размеры области вокруг игрока
    region_size = 2 * radius + 1

    # Извлекаем область карты вокруг игрока с помощью lax.dynamic_slice
    region = lax.dynamic_slice(
        game_map,
        start_indices=(x - radius, y - radius),
        slice_sizes=(region_size, region_size)
    )


    # Размер квадрата, добавляем +2 для проверки краёв
    square_size = distance*2 + 3

    def check_square(i, j):
    # Извлекаем текущий квадрат с помощью lax.dynamic_slice
        square = lax.dynamic_slice(
            region,
            start_indices=(i, j),
            slice_sizes=(square_size, square_size)
        )

        # Находится ли target в центре?
        target_mask = square[round(square_size / 2), round(square_size / 2)] == target_object_name

        # Маска для object_name (в зависимости от стороны на расстоянии distance)
        object_mask = lax.switch(side,
            [
                lambda: square[round(square_size / 2), distance] == object_name,  # Справа
                lambda: square[round(square_size / 2), 0] == object_name,  # Слева
                lambda: square[distance, round(square_size / 2)] == object_name,  # Сверху
                lambda: square[0, round(square_size / 2)] == object_name  # Снизу
            ]
        )

        # Проверка, что и target, и объект находятся на своих местах
        return target_mask & object_mask    

    # Проходим по квадратикам вокруг игрока
    result = jnp.any(jax.vmap(lambda i, j: check_square(i, j))(jnp.arange(0, region.shape[0] - square_size + 1),
                                                                jnp.arange(0, region.shape[1] - square_size + 1)))

    return result

def move_to(game_data, side):
    """
    Check if the player is significantly on one of the sides (North, West, East, South) of the map.
    A player is considered to be on a side if they are at least 20 units away from the center of the map
    in the respective direction.

    Args:
    - game_data (GameData): The game data object containing the player's positions and map information.
    - side (int): The side to check. Encoded as:
                  0 = North, 1 = West, 2 = East, 3 = South.

    Returns:
    - bool: True if the player is significantly on the specified side of the map, False otherwise.
    """

    game_map = game_data.states[0].map.game_map[0]  # Assume 100x100 JAX array
    player_position = game_data.states[0].variables.player_position

    if player_position is None:
        return False

    x, y = player_position
    center_x, center_y = game_map.shape[0] // 2, game_map.shape[1] // 2  # Center of the map

    def check_position_in_side(side, x, y, center_x, center_y):
        # Check if the player is significantly on one of the sides (20 units away from the center)
        return lax.cond(
            side == 0, lambda _: (x < center_x - 20),  # North
            lambda _: lax.cond(
                side == 1, lambda _: (y < center_y - 20),  # West
                lambda _: lax.cond(
                    side == 2, lambda _: (y > center_y + 20),  # East
                    lambda _: (x > center_x + 20)  # South
                )
            ),
            operand=None
        )

    # Check if the player's position is in the specified side of the map
    is_in_side = check_position_in_side(side, x, y, center_x, center_y)

    return is_in_side
