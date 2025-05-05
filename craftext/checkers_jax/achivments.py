import jax
import jax.numpy as jnp
from craftext.adapters.state_adapter import GameData
from craftext.checkers_jax.target_state import Achievements, AchievementState

def checker_acvievments(game_data: GameData,  target_state: Achievements) -> jax.Array:
    
    achievement_mask  = target_state.achievement_mask
    
    # return jax.lax.select(target_state.need_to_achieve, 
    #                conditional_achivments(game_data, achievement_mask),
    #                jnp.array(False))
    return conditional_achivments(game_data, achievement_mask)

def conditional_achivments(gd: GameData, achievement_mask):
    # 0) Убедимся, что маска — это JAX‑массив
    mask = jnp.array(achievement_mask, dtype=jnp.int32)

    current_state      = gd.states[0]
    state_achievements = current_state.achievements.achievements  # jnp.array of 0/1

    # 1) там, где mask == NEED, но ещё не достигнуто (state==0) → ошибка
    must_achieve     = (mask == AchievementState.NEED_TO_ACHIEVE) & (state_achievements == 0)
    # 2) там, где mask == AVOID, но уже достигнуто (state==1) → ошибка
    must_not_achieve = (mask == AchievementState.AVOID_TO_ACHIEVE) & (state_achievements == 1)

    # если хоть раз ошибились — fail=True
    fail_condition = jnp.any(must_achieve) | jnp.any(must_not_achieve)

    # True только когда нет ни одного fail
    return jnp.logical_not(fail_condition)
# def conditional_achivments(gd: GameData, achievement_mask) -> jax.Array:
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
#     current_state = gd.states[0]
#     state_achievements = current_state.achievements.achievements 
#     must_achieve = jnp.logical_and(achievement_mask == AchievementState.NEED_TO_ACHIEVE, state_achievements == 0)
#     must_not_achieve = jnp.logical_and(achievement_mask == AchievementState.AVOID_TO_ACHIEVE, state_achievements == 1)

#     # If any must_achieve or must_not_achieve fails, return False
#     fail_condition = jnp.any(must_achieve) | jnp.any(must_not_achieve)

#     return jnp.logical_not(fail_condition)


