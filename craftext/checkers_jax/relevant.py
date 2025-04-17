import jax
from jax import (
    numpy as jnp,
    lax
)
from craftext.adapters.state_adapter import GameData
from craftext.checkers_jax.target_state import TargetState

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

    region_masked = jnp.where(mask, region, -1)
    return region_masked

def place_object_relevant_to(game_data: GameData, target_state: TargetState) -> jax.Array:
    """
    Check if the object is placed at a specific side (right, left, top, or bottom) and distance from the target_object_name
    within the area around the player.
    """

    # Если условие не нужно достигать, возвращаем False
    object_name = target_state.Localization_placing.object_name
    target_object_name = target_state.Localization_placing.target_object_name
    side = target_state.Localization_placing.side
    distance = target_state.Localization_placing.distance
    
    game_map = game_data.states[0].map.game_map
    player_position = game_data.states[0].variables.player_position
    x, y = player_position

    # Фиксированный радиус области вокруг игрока
    radius = 5
    region_size = 2 * 20 + 1

    # Извлекаем область карты вокруг игрока через lax.dynamic_slice
    region = safe_dynamic_slice(game_map, x, y, radius, region_size)
    
    # Размер квадрата для проверки; +2 для проверки краёв
    square_size = 10 * 2 + 1
    
    # Определяем сканирование по всем квадратам внутри region:
    n_rows = region.shape[0] - square_size + 1
    n_cols = region.shape[1] - square_size + 1

    def check_square(i, j):
        # Извлекаем квадрат размером (square_size, square_size)
        square = lax.dynamic_slice(
            region,
            start_indices=(i, j),
            slice_sizes=(square_size, square_size)
        )
        center = 10  # поскольку square_size = 2*distance+1, центр = distance
        target_mask = square[center, center] == target_object_name
        object_mask = lax.switch(side,
            [
                lambda: square[center, -1] == object_name,  # Справа
                lambda: square[center,  0] == object_name,  # Слева
                lambda: square[-1, center] == object_name,  # Сверху
                lambda: square[0, center]  == object_name   # Снизу
            ]
        )
        return target_mask & object_mask

    def row_loop(i, carry):
        def col_loop(j, inner):
            valid = check_square(i, j)
            return jnp.logical_or(inner, valid)
        row_result = jax.lax.fori_loop(0, n_cols, col_loop, False)
        return jnp.logical_or(carry, row_result)

    overall_result = jax.lax.fori_loop(0, n_rows, row_loop, False)
    return overall_result

def move_to(game_data, side) -> jax.Array:
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
        return jnp.array(False)

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
