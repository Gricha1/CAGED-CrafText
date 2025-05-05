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


from functools import partial
# Фиксированный радиус, которым мы захватываем все возможные distance ≤ MAX_RADIUS
MAX_RADIUS = 5
REGION_SIZE = 2 * MAX_RADIUS + 1  # статичный

@jax.jit
def place_object_relevant_to(
    game_data, 
    object_name: str, 
    target_object_name: str, 
    side: int,      # tracer-скаляр: 0=Right,1=Left,2=Top,3=Bottom
    distance: int   # tracer-скаляр, ≤ MAX_RADIUS
) -> jax.Array:
    # 1) Собираем единый REGION вокруг игрока размером REGION_SIZE×REGION_SIZE
    x, y = game_data.states[0].variables.player_position
    padded_map = jnp.pad(
        game_data.states[0].map.game_map,
        ((MAX_RADIUS, MAX_RADIUS), (MAX_RADIUS, MAX_RADIUS)),
        constant_values=-1  # любая метка за границей
    )
    region = lax.dynamic_slice(
        padded_map,
        (x, y),
        (REGION_SIZE, REGION_SIZE)
    )  # → shape [REGION_SIZE, REGION_SIZE]

    # 2) Делаем булевы карты для цели и объекта
    #    (можно сразу сравнить с .value, или если у вас int-коды — без .value)
    tgt_mask = (region == target_object_name)
    obj_mask = (region == object_name)

    # 3) Чтобы сдвинуть obj_mask на (side, distance), ещё раз паддим region/obj_mask
    #    на MAX_RADIUS ║ MAX_RADIUS, и будем «секурно» брать кусок REGION_SIZE×REGION_SIZE
    padded_obj = jnp.pad(
        obj_mask,
        ((MAX_RADIUS, MAX_RADIUS), (MAX_RADIUS, MAX_RADIUS)),
        constant_values=False
    )

    # 4) Вычисляем динамический сдвиг в координатах padded_obj
    #    Право  (0):  di= 0, dj=+distance
    #    Лево  (1):  di= 0, dj=-distance
    #    Верх  (2):  di=-distance, dj=0
    #    Низ   (3):  di=+distance, dj=0
    di = jnp.where(side==2, -distance,
         jnp.where(side==3,  distance, 0))
    dj = jnp.where(side==0,  distance,
         jnp.where(side==1, -distance, 0))

    # Стартовый индекс в padded_obj: центр + (di,dj)
    start_i = MAX_RADIUS + di
    start_j = MAX_RADIUS + dj

    # 5) Единичный dynamic_slice, **фиксированный** REGION_SIZE×REGION_SIZE
    shifted_obj = lax.dynamic_slice(
        padded_obj,
        (start_i, start_j),
        (REGION_SIZE, REGION_SIZE)
    )

    # 6) Проверяем, есть ли позиция (i,j), где одновременно tgt_mask[i,j] и shifted_obj[i,j]
    hit = tgt_mask & shifted_obj

    # 7) Нужен ли `need_to_achieve`? Если да, поднимайте его снаружи через select()  
    return jnp.any(hit)


# def place_object_relevant_to(game_data: GameData, object_name: int, target_object_name: int, side: int, distance: int) -> jax.Array:

    
#     game_map = game_data.states[0].map.game_map
#     player_position = game_data.states[0].variables.player_position
#     x, y = player_position

#     radius = 5
#     region_size = 2 * 5 + 1

#     region = safe_dynamic_slice(game_map, x, y, radius, region_size)
    
#     square_size = 5 * 2 + 2
    
#     n_rows = region.shape[0] - square_size + 1
#     n_cols = region.shape[1] - square_size + 1

#     def check_square(i, j):
#         square = lax.dynamic_slice(
#             region,
#             start_indices=(i, j),
#             slice_sizes=(square_size, square_size)
#         )
#         center = 5  
#         target_mask = square[center, center] == target_object_name
#         object_mask = lax.switch(side,
#             [
#                 lambda: square[center, -1] == object_name,  
#                 lambda: square[center,  0] == object_name,  
#                 lambda: square[-1, center] == object_name,  
#                 lambda: square[0, center]  == object_name   
#             ]
#         )
#         return target_mask & object_mask

#     def row_loop(i, carry):
#         def col_loop(j, inner):
#             valid = check_square(i, j)
#             return jnp.logical_or(inner, valid)
#         row_result = jax.lax.fori_loop(0, n_cols, col_loop, False)
#         return jnp.logical_or(carry, row_result)

#     overall_result = jax.lax.fori_loop(0, n_rows, row_loop, False)
#     return overall_result

# def move_to(game_data, side) -> jax.Array:


#     game_map = game_data.states[0].map.game_map[0]  
#     player_position = game_data.states[0].variables.player_position

#     if player_position is None:
#         return jnp.array(False)

#     x, y = player_position
#     center_x, center_y = game_map.shape[0] // 2, game_map.shape[1] // 2  

#     def check_position_in_side(side, x, y, center_x, center_y):
#         return lax.cond(
#             side == 0, lambda _: (x < center_x - 20),  
#             lambda _: lax.cond(
#                 side == 1, lambda _: (y < center_y - 20),  
#                 lambda _: lax.cond(
#                     side == 2, lambda _: (y > center_y + 20),  
#                     lambda _: (x > center_x + 20)  
#                 )
#             ),
#             operand=None
#         )

#     is_in_side = check_position_in_side(side, x, y, center_x, center_y)

#     return is_in_side
