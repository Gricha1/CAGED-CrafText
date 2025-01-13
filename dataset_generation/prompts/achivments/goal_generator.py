import pandas as pd
import random

# ---------------------------
# Constants
# ---------------------------

ACHIEVEMENTS = [
    'COLLECT_WOOD', 'PLACE_TABLE', 'EAT_COW', 'COLLECT_SAPLING',
    'COLLECT_DRINK', 'MAKE_WOOD_PICKAXE', 'MAKE_WOOD_SWORD', 
    'PLACE_PLANT', 'DEFEAT_ZOMBIE', 'COLLECT_STONE', 'PLACE_STONE',
    'EAT_PLANT', 'DEFEAT_SKELETON', 'MAKE_STONE_PICKAXE', 'MAKE_STONE_SWORD',
    'WAKE_UP', 'PLACE_FURNACE', 'COLLECT_COAL', 'COLLECT_IRON', 
    'COLLECT_DIAMOND', 'MAKE_IRON_PICKAXE', 'MAKE_IRON_SWORD'
]

OBJECT_TO_ACHIEVEMENT = {
    'tree': 'COLLECT_WOOD',
    'water': 'COLLECT_DRINK',
    'drink': 'COLLECT_DRINK',
    'grass': 'PLACE_PLANT',
    'sapling': 'COLLECT_SAPLING',
    'plant': 'PLACE_PLANT',
    'stone': 'COLLECT_STONE',
    'table': 'PLACE_TABLE',
    'furnace': 'PLACE_FURNACE',
    'coal': 'COLLECT_COAL',
    'iron': 'COLLECT_IRON',
    'diamond': 'COLLECT_DIAMOND',
    'wood': 'COLLECT_WOOD',
    'wood_pickaxe': 'MAKE_WOOD_PICKAXE',
    'wood_sword': 'MAKE_WOOD_SWORD',
    'stone_pickaxe': 'MAKE_STONE_PICKAXE',
    'stone_sword': 'MAKE_STONE_SWORD',
    'iron_pickaxe': 'MAKE_IRON_PICKAXE',
    'iron_sword': 'MAKE_IRON_SWORD'
}

ACHIEVEMENT_TO_OBJECT = {v: k for k, v in OBJECT_TO_ACHIEVEMENT.items()}

# ---------------------------
# Data Loader
# ---------------------------

def load_independent_pairs(filepath: str) -> pd.DataFrame:
    """
    Load independent pairs from a CSV file.
    """
    return pd.read_csv(filepath)

# ---------------------------
# Goal Generation
# ---------------------------

def generate_goal(independent_pairs: pd.DataFrame) -> str:
    """
    Generates a goal configuration based on specified rules.
    """
    num_objects = random.choice([2, 3])
    with_exclusion = random.choice([True, False])
    
    selected_achievement = random.choice(ACHIEVEMENTS)
    selected_object = ACHIEVEMENT_TO_OBJECT.get(selected_achievement)
    selected_objects = [selected_achievement]
    
    if num_objects == 2:
        goal = _generate_two_object_goal(
            selected_achievement, selected_object, with_exclusion, independent_pairs
        )
    else:
        goal = _generate_three_object_goal(
            selected_achievement, selected_object, with_exclusion, independent_pairs
        )
    
    return goal

def _generate_two_object_goal(selected_achievement, selected_object, with_exclusion, independent_pairs):
    if with_exclusion and selected_object:
        independent_options = independent_pairs[independent_pairs['Object 1'] == selected_object]['Object 2'].tolist()
        if independent_options:
            excluded_object = random.choice(independent_options)
            excluded_achievement = OBJECT_TO_ACHIEVEMENT.get(excluded_object)
            if excluded_achievement:
                return f"{selected_achievement} AND NOT {excluded_achievement}"
    second_object = random.choice(ACHIEVEMENTS)
    return f"{selected_achievement} AND {second_object}"

def _generate_three_object_goal(selected_achievement, selected_object, with_exclusion, independent_pairs):
    second_object = random.choice(ACHIEVEMENTS)
    third_object = random.choice(ACHIEVEMENTS)
    
    if with_exclusion and selected_object:
        independent_options = independent_pairs[independent_pairs['Object 1'] == selected_object]['Object 2'].tolist()
        if independent_options:
            excluded_object = random.choice(independent_options)
            excluded_achievement = OBJECT_TO_ACHIEVEMENT.get(excluded_object)
            if excluded_achievement:
                return f"{selected_achievement} AND {second_object} AND NOT {excluded_achievement}"
    
    return f"{selected_achievement} AND {second_object} AND {third_object}"

# ---------------------------
# Goal Examples Generator
# ---------------------------

def generate_example_goals(num_goals: int = 100, difficulty='EASY'):
    """
    Generate a list of unique example goals.
    """
    independent_pairs = load_independent_pairs('prompts/achivments/independent_pairs.csv')
    example_goals = set()
    make_synonims_to = []
    while len(example_goals) < num_goals:
        goal = generate_goal(independent_pairs)
        goal_parts = goal.split(" AND ")
        random.shuffle(goal_parts)
        shuffled_goal = " AND ".join(goal_parts)
        if shuffled_goal not in example_goals:
            synonums_to=f"WHEN GENERATING PARAPHRASES, USE SYNONYMS OR ALTERNATIVE EXPRESSIONS \
                        FOR THE TERM '{goal_parts}' TO MAKE THE TEXT MORE DIVERSE AND NATURAL."
            make_synonims_to.append(synonums_to)
        example_goals.add(shuffled_goal)
    
    return list(example_goals), make_synonims_to

# ---------------------------
# Main Entry Point
# ---------------------------

if __name__ == "__main__":
    independent_pairs = load_independent_pairs('prompts/achivments/independent_pairs.csv')
    goals = generate_example_goals(independent_pairs, num_goals=100)
    
    for i, goal in enumerate(goals):
        print(f"{i}: {goal}")
