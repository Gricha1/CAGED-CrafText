import jax.numpy as jnp
import jax.lax as lax
import jax
from flax.struct import dataclass

from craftext.adapters.state_adapter import GameData
from craftext.checkers_jax.target_state import TargetState, BuildStarState, BuildStarState
from functools import partial


import jax
import jax.numpy as jnp

def check_cross(region: jax.Array, stone_index: int, size: int, cross_type: int) -> jax.Array:
    center = size // 2

    def compute_straight(_):
        return (jnp.all(region[center, :] == stone_index) &
                jnp.all(region[:, center] == stone_index))

    def compute_diagonal(_):
        return (jnp.all(jnp.diag(region) == stone_index) &
                jnp.all(jnp.diag(jnp.fliplr(region)) == stone_index))


    def compute_size3(_):
        straight = compute_straight(None)
        diagonal = compute_diagonal(None)
        return straight | diagonal


    def compute_size5_7(_):
        straight = compute_straight(None)
        diagonal = compute_diagonal(None)
        return straight | diagonal | (straight & diagonal)

    base_result = jax.lax.cond(
        size == 3,
        compute_size3,
        lambda _: jax.lax.cond(
            (size == 5) | (size == 7),
            compute_size5_7,
            lambda _: jnp.array(False), 
            operand=None
        ),
        operand=None
    )


    final_result = jax.lax.cond(
        cross_type == 0,
        lambda _: compute_straight(None),
        lambda _: jax.lax.cond(
            cross_type == 1,
            lambda _: compute_diagonal(None),
            lambda _: base_result,
            operand=None
        ),
        operand=None
    )

    return final_result


@dataclass
class Carry:
    region: jax.Array
    stone_index: int
    region_size: int
    size: int
    cross_type: int

def scan_cross_function(carry: Carry, x):
    i, j = x // carry.region_size, x % carry.region_size
    
    
    condition = jnp.logical_or(i + carry.size > carry.region_size,
                            j + carry.size > carry.region_size)
    
    def branch_true(_):
        sub_region= carry.region[i:i+carry.size, j:j+carry.size]
        is_cross = check_cross(sub_region, carry.stone_index, carry.size, carry.cross_type)
        
        return is_cross

    return carry, lax.cond(condition, branch_true, lambda _: False, operand=None)

def is_cross_formed(game_data: GameData,  target_state: BuildStarState) -> jax.Array:
    
    block_index = target_state.block_type
    radius = target_state.radius
    size = target_state.size
    cross_type = target_state.cross_type

    return jax.lax.select(target_state.need_to_achieve, 
                   cross_checker(10, 7, game_data, block_index, radius, size, cross_type),
                   jnp.array(False))

# # Пределы (выберите по самому большому сценарию)
# MAX_RADIUS = 10
# MAX_SIZE   = 7  # поддерживаем крестики вплоть до size=7

@partial(jax.jit, static_argnums=(0,1))
def cross_checker(
    max_radius: int,
    max_size:   int,
    game_data,
    block_index: int,    # tracer-скаляр
    cross_type:  int,    # tracer-скаляр
    radius:      int,    # tracer-скаляр, <= max_radius
    size:        int     # tracer-скаляр, <= max_size и нечётное
):
    """
    Ищет крестик типа cross_type и длины size
    из блоков block_index в области radius вокруг игрока.
    """

    # --- A) вырезаем всегда фиксированный регион (2*max_radius+1)^2 вокруг игрока ---
    x, y = game_data.states[0].variables.player_position
    R   = max_radius
    FULL = 2*R + 1
    padded = jnp.pad(
        game_data.states[0].map.game_map,
        ((R, R), (R, R)),
        constant_values=-1
    )
    region_full = lax.dynamic_slice(padded, (x, y), (FULL, FULL))  # [FULL, FULL]

    # --- B) маскируем вне реального radius ---
    coords = jnp.arange(-R, R+1)                         # shape [FULL]
    mask1d = jnp.abs(coords) <= radius                   # tracer → shape [FULL]
    mask2d = mask1d[:, None] & mask1d[None, :]           # [FULL, FULL]
    region = jnp.where(mask2d, region_full, -1)

    # --- C) бинарная карта для блока ---
    B = (region == block_index).astype(jnp.float32)[None, None, ...]

    # --- D) строим динамические фильтры в пространстве max_size×max_size ---
    S = max_size
    C = S // 2   # центр фильтра

    idxs = jnp.arange(S)  # static arange от 0 до S-1

    # 1) маска «раста» вдоль одной оси длины size
    half = size // 2      # tracer
    start = C - half      # tracer
    end   = start + size  # tracer

    # mask_range[p] = True iff start <= p < end
    mask_range = (idxs >= start) & (idxs < end)  # shape [S], tracer

    # строим фильтры
    # горизонтальный: строка = C, столбцы ∈ mask_range
    row_idx = idxs[:, None]      # [S,1]
    col_idx = idxs[None, :]      # [1,S]
    filt_h = (row_idx == C) & mask_range[None, :]  # [S,S]
    # вертикальный: столбец = C, строки ∈ mask_range
    filt_v = (col_idx == C) & mask_range[:, None]  # [S,S]
    # диагонали:
    filt_d1 = (row_idx == col_idx) & mask_range[:, None] & mask_range[None, :]
    filt_d2 = (row_idx + col_idx == 2*C) & mask_range[:, None] & mask_range[None, :]

    # приводим к float
    kh = filt_h.astype(jnp.float32)
    kv = filt_v.astype(jnp.float32)
    kd1 = filt_d1.astype(jnp.float32)
    kd2 = filt_d2.astype(jnp.float32)

    # --- E) свёртки (без изменения формы) ---
    conv = partial(lax.conv_general_dilated,
                   window_strides=(1,1),
                   padding="VALID",
                   dimension_numbers=("NCHW","OIHW","NCHW"))

    h_out  = conv(B, kh [None, None])[0,0]  # [FULL-S+1, FULL-S+1]
    v_out  = conv(B, kv [None, None])[0,0]
    d1_out = conv(B, kd1[None, None])[0,0]
    d2_out = conv(B, kd2[None, None])[0,0]

    # --- F) проверяем полные совпадения длины size ---
    straight = (h_out  == size) & (v_out  == size)
    diagonal = (d1_out == size) & (d2_out == size)

    # --- G) выбор по cross_type ---
    mask = lax.cond(
        cross_type == 0,
        lambda _: straight,
        lambda _: lax.cond(
            cross_type == 1,
            lambda _: diagonal,
            lambda _: straight | diagonal,
            operand=None
        ),
        operand=None
    )

    # --- H) есть ли True? ---
    return jnp.any(mask)

# def cross_checker(game_data: GameData, block_index: int, radius: int, size: int, cross_type: int) -> jax.Array:

#     game_map = game_data.states[0].map.game_map
    
#     player_position = game_data.states[0].variables.player_position
    
#     x, y = player_position
#     region_size = 2 * radius + 1
#     region = lax.dynamic_slice(
#         game_map,
#         start_indices=(x - radius, y - radius),
#         slice_sizes=(region_size, region_size)
#     )

#     indixes = jnp.arange(region_size*region_size)
#     carry = Carry(region=region,
#             stone_index=block_index,
#             region_size=region_size,
#             size=size,
#             cross_type=cross_type)
#     _, crosses = lax.scan(scan_cross_function, carry, indixes)
    
#     return jnp.any(crosses)

# # 89.169.171.236

