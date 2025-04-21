import jax
from jax import (
    numpy as jnp,
    lax
)
from craftext.adapters.state_adapter import GameData
from craftext.checkers_jax.target_state import TargetState, LocalizaPlacingState
from functools import partial


@partial(jax.jit, static_argnames=['max_radius'])
def safe_dynamic_slice(game_map, x, y, radius, max_radius):
    full_region_size = 2 * max_radius + 1

    x_padded = x + max_radius
    y_padded = y + max_radius

    region = lax.dynamic_slice(
        game_map,
        start_indices=(x_padded - max_radius, y_padded - max_radius),
        slice_sizes=(full_region_size, full_region_size)
    )

    coord_range = jnp.arange(full_region_size) - max_radius
    mask_x = jnp.abs(coord_range) <= radius
    mask_y = mask_x[:, None]
    mask = mask_x & mask_y

    region_masked = jnp.where(mask, region, -1)
    return region_masked


def cheker_localization(game_data: GameData,  target_state: LocalizaPlacingState) -> jax.Array:
    
    object_name = target_state.object_name
    target_object_name = target_state.target_object_name 
    side = target_state.side 
    distance = target_state.distance

    # return jax.lax.select(target_state.need_to_achieve, 
    #                localization_checker(game_data=game_data, object_name=object_name, target_object_name=target_object_name, side=side, distance=distance),
    #                jnp.array(False))
    return place_object_relevant_to(game_data=game_data, object_name=object_name, target_object_name=target_object_name, side=side, distance=distance)

def place_object_relevant_to(game_data: GameData, object_name: int, target_object_name: int, side: int, distance: int) -> jax.Array:

    
    game_map = game_data.states[0].map.game_map
    player_position = game_data.states[0].variables.player_position
    x, y = player_position

    radius = 5
    region_size = 2 * 5 + 1

    region = safe_dynamic_slice(game_map, x, y, radius, region_size)
    
    square_size = 5 * 2 + 2
    
    n_rows = region.shape[0] - square_size + 1
    n_cols = region.shape[1] - square_size + 1

    def check_square(i, j):
        square = lax.dynamic_slice(
            region,
            start_indices=(i, j),
            slice_sizes=(square_size, square_size)
        )
        center = 5  
        target_mask = square[center, center] == target_object_name
        object_mask = lax.switch(side,
            [
                lambda: square[center, -1] == object_name,  
                lambda: square[center,  0] == object_name,  
                lambda: square[-1, center] == object_name,  
                lambda: square[0, center]  == object_name   
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


    game_map = game_data.states[0].map.game_map[0]  
    player_position = game_data.states[0].variables.player_position

    if player_position is None:
        return jnp.array(False)

    x, y = player_position
    center_x, center_y = game_map.shape[0] // 2, game_map.shape[1] // 2  

    def check_position_in_side(side, x, y, center_x, center_y):
        return lax.cond(
            side == 0, lambda _: (x < center_x - 20),  
            lambda _: lax.cond(
                side == 1, lambda _: (y < center_y - 20),  
                lambda _: lax.cond(
                    side == 2, lambda _: (y > center_y + 20),  
                    lambda _: (x > center_x + 20)  
                )
            ),
            operand=None
        )

    is_in_side = check_position_in_side(side, x, y, center_x, center_y)

    return is_in_side
