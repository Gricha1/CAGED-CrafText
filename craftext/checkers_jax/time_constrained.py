import jax.numpy as jnp
import jax.lax as lax
import jax
from craftext.adapters.state_adapter import GameData
from craftext.checkers_jax.target_state import TimeCosntrainedPlacmentState



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

    coord_range = jnp.arange(full_region_size) - max_radius
    mask_x = jnp.abs(coord_range) <= radius
    mask_y = mask_x[:, None]
    mask = mask_x & mask_y

    region_masked = jnp.where(mask, region, 0)
    return region_masked

def checker_time_placement(game_data: GameData,  target_state: TimeCosntrainedPlacmentState) -> jax.Array:
    
    block_index = target_state.block_type
    radius = target_state.radius
    time_state = target_state.time_state
    # return jax.lax.select(target_state.need_to_achieve, 
    #                time_place_checker(game_data, block_index, radius, time_state),
    #                jnp.array(False))
    return at_time_block_placed(game_data, block_index, radius, time_state)

def at_time_block_placed(game_data: GameData, block_index, radius, time_state) -> jax.Array:
    
    x, y = game_data.states[0].variables.player_position
    region = safe_dynamic_slice(game_data.states[0].map.game_map, x, y, radius, 5)
    in_range = jnp.abs(game_data.states[0].variables.light_level - jax.lax.clamp(0, time_state, 1)) <= 0.2
    return in_range & (region == block_index).any()
