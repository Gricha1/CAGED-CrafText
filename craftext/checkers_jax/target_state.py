from flax import struct
import jax.numpy as jnp
from craftext.scenarios.constants import Achievement, AchievementState


@struct.dataclass
class Achievements:
    achievement_mask: jnp.ndarray


@struct.dataclass
class TargetState:
    achievements: Achievements



    # collect_wood: int = AchievementState.NOT_MATTER.value
    # place_table: int = AchievementState.NOT_MATTER.value
    # eat_cow: int = AchievementState.NOT_MATTER.value
    # collect_sapling: int = AchievementState.NOT_MATTER.value
    # collect_drink: int = AchievementState.NOT_MATTER.value
    # make_wood_pickaxe: int = AchievementState.NOT_MATTER.value
    # make_wood_sword: int = AchievementState.NOT_MATTER.value
    # place_plant: int = AchievementState.NOT_MATTER.value
    # defeat_zombie: int = AchievementState.NOT_MATTER.value
    # collect_stone: int = AchievementState.NOT_MATTER.value
    # place_stone: int = AchievementState.NOT_MATTER.value
    # eat_plant: int = AchievementState.NOT_MATTER.value
    # defeat_skeleton: int = AchievementState.NOT_MATTER.value
    # make_stone_pickaxe: int = AchievementState.NOT_MATTER.value
    # make_stone_sword: int = AchievementState.NOT_MATTER.value
    # wake_up: int = AchievementState.NOT_MATTER.value
    # place_furnace: int = AchievementState.NOT_MATTER.value
    # collect_coal: int = AchievementState.NOT_MATTER.value
    # collect_iron: int = AchievementState.NOT_MATTER.value
    # collect_diamond: int = AchievementState.NOT_MATTER.value
    # make_iron_pickaxe: int = AchievementState.NOT_MATTER.value
    # make_iron_sword: int = AchievementState.NOT_MATTER.value
    # make_arrow: int = AchievementState.NOT_MATTER.value
    # make_torch: int = AchievementState.NOT_MATTER.value
    # place_torch: int = AchievementState.NOT_MATTER.value
    # collect_sapphire: int = AchievementState.NOT_MATTER.value
    # collect_ruby: int = AchievementState.NOT_MATTER.value
    # make_diamond_pickaxe: int = AchievementState.NOT_MATTER.value
    # make_diamond_sword: int = AchievementState.NOT_MATTER.value
    # make_iron_armour: int = AchievementState.NOT_MATTER.value
    # make_diamond_armour: int = AchievementState.NOT_MATTER.value
    # enter_gnomish_mines: int = AchievementState.NOT_MATTER.value
    # enter_dungeon: int = AchievementState.NOT_MATTER.value
    # enter_sewers: int = AchievementState.NOT_MATTER.value
    # enter_vault: int = AchievementState.NOT_MATTER.value
    # enter_troll_mines: int = AchievementState.NOT_MATTER.value
    # enter_fire_realm: int = AchievementState.NOT_MATTER.value
    # enter_ice_realm: int = AchievementState.NOT_MATTER.value
    # enter_graveyard: int = AchievementState.NOT_MATTER.value
    # defeat_gnome_warrior: int = AchievementState.NOT_MATTER.value
    # defeat_gnome_archer: int = AchievementState.NOT_MATTER.value
    # defeat_orc_solider: int = AchievementState.NOT_MATTER.value
    # defeat_orc_mage: int = AchievementState.NOT_MATTER.value
    # defeat_lizard: int = AchievementState.NOT_MATTER.value
    # defeat_kobold: int = AchievementState.NOT_MATTER.value
    # defeat_knight: int = AchievementState.NOT_MATTER.value
    # defeat_archer: int = AchievementState.NOT_MATTER.value
    # defeat_troll: int = AchievementState.NOT_MATTER.value
    # defeat_deep_thing: int = AchievementState.NOT_MATTER.value
    # defeat_pigman: int = AchievementState.NOT_MATTER.value
    # defeat_fire_elemental: int = AchievementState.NOT_MATTER.value
    # defeat_frost_troll: int = AchievementState.NOT_MATTER.value
    # defeat_ice_elemental: int = AchievementState.NOT_MATTER.value
    # damage_necromancer: int = AchievementState.NOT_MATTER.value
    # defeat_necromancer: int = AchievementState.NOT_MATTER.value
    # eat_bat: int = AchievementState.NOT_MATTER.value
    # eat_snail: int = AchievementState.NOT_MATTER.value
    # find_bow: int = AchievementState.NOT_MATTER.value
    # fire_bow: int = AchievementState.NOT_MATTER.value
    # learn_fireball: int = AchievementState.NOT_MATTER.value
    # cast_fireball: int = AchievementState.NOT_MATTER.value
    # learn_iceball: int = AchievementState.NOT_MATTER.value
    # cast_iceball: int = AchievementState.NOT_MATTER.value
    # open_chest: int = AchievementState.NOT_MATTER.value
    # drink_potion: int = AchievementState.NOT_MATTER.value
    # enchant_sword: int = AchievementState.NOT_MATTER.value
    # enchant_armour: int = AchievementState.NOT_MATTER.value
    # smth: int = AchievementState.NOT_MATTER.value
    # end: int = AchievementState.NOT_MATTER.value

# @struct.dataclass
# class TargetState:
#     achievements: Achievements


# --- Запуск тестов ---
if __name__ == "__main__":
    existing_achievements = Achievements()
    target_state = create_target_state()

    N = 100000  # Количество повторов

    # Тест 1: Создание нового объекта Achievements
    start = time.time()
    for _ in range(N):
        create_new_achievements()
    print(f"New Achievements: {time.time() - start:.6f} sec")

    # Тест 2: Использование replace()
    start = time.time()
    for _ in range(N):
        replace_achievements(existing_achievements)
    print(f"Replace Achievements: {time.time() - start:.6f} sec")

    # Тест 3: Создание нового TargetState
    start = time.time()
    for _ in range(N):
        create_target_state()
    print(f"New TargetState: {time.time() - start:.6f} sec")

    # Тест 4: JIT-компилированная проверка
    check_achievements(target_state, target_state)  # Прогрев JIT
    start = time.time()
    for _ in range(N):
        check_achievements(target_state, target_state)
    print(f"JIT check Achievements: {time.time() - start:.6f} sec")
