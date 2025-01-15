import jax
import jax.numpy as jnp

def conditional_achivments(gd, vector_to_achieve: jax.Array) -> jax.Array:
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
    # Conditions for must-achieve (1) and must-not-achieve (-1)
    must_achieve = jnp.logical_and(vector_to_achieve == 1, state_achievements != 1)
    must_not_achieve = jnp.logical_and(vector_to_achieve == -1, state_achievements!= 0)

    # If any must_achieve or must_not_achieve fails, return False
    fail_condition = jnp.any(must_achieve) | jnp.any(must_not_achieve)

    return jnp.logical_not(fail_condition)
