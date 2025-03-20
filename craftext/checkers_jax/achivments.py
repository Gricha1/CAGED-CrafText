import jax
import jax.numpy as jnp

from craftext.checkers.base_functions.state_adapter import GameData
from craftext.checkers_jax.target_state import TargetState


def conditional_achivments(gd: GameData, target_state: TargetState) -> jax.Array:
    """
    Parameters:
        gd.state.achievements - jnp.array: Boolean vector (0 = not achieved, 1 = achieved)
        vector_to_achieve - jnp.array: Vector with values 1, 0, -1
            1 = must be achieved
            0 = doesn't matter
            -1 = must not be achieved
    Returns:
        bool - True if all conditions are met, False otherwise
    """
    current_state = gd.states[1]
    state_achievements = current_state.achievements.achievements 
    vector_to_achieve = target_state.achievements.achievement_mask
    # Conditions for must-achieve (1) and must-not-achieve (-1)
    must_achieve = jnp.logical_and(vector_to_achieve == 1, state_achievements != 1)
    must_not_achieve = jnp.logical_and(vector_to_achieve == -1, state_achievements!= 0)

    # If any must_achieve or must_not_achieve fails, return False
    fail_condition = jnp.any(must_achieve) | jnp.any(must_not_achieve)

    return jnp.logical_not(fail_condition)

# def conditional_achivments(gd: GameData, vector_to_achieve: jax.Array) -> jax.Array:
#     """
#     Parameters:
#         gd.state.achievements - jnp.array: Boolean vector (0 = not achieved, 1 = achieved)
#         vector_to_achieve - jnp.array: Vector with values 1, 0, -1
#             1 = must be achieved
#             0 = doesn't matter
#             -1 = must not be achieved
#     Returns:
#         bool - True if all conditions are met, False otherwise
#     """
#     current_state = gd.states[1]
#     state_achievements = current_state.achievements.achievements 
#     # Conditions for must-achieve (1) and must-not-achieve (-1)
#     must_achieve = jnp.logical_and(vector_to_achieve == 1, state_achievements != 1)
#     must_not_achieve = jnp.logical_and(vector_to_achieve == -1, state_achievements!= 0)

#     # If any must_achieve or must_not_achieve fails, return False
#     fail_condition = jnp.any(must_achieve) | jnp.any(must_not_achieve)

#     return jnp.logical_not(fail_condition)
