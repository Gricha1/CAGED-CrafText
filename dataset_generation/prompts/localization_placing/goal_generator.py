import random
def generate_example_goals(num_goals: int, difficulty="EASY") -> list[str]:
    """
    Generate a list of unique text-based goals for placing objects relative to targets.
    
    Args:
        num_goals (int): Number of goals to generate.
    
    Returns:
        list[str]: List of unique place-object goals.
    """
    place_objects = {
        'EASY': ['STONE', 'CRAFTING_TABLE', 'PLANT', 'FURNACE'],
        'MEDIUM': ['STONE', 'CRAFTING_TABLE', 'PLANT', 'FURNACE', 'ENCHANTMENT_TABLE_FIRE', 'ENCHANTMENT_TABLE_ICE']
    }
    target_objects = {
        'EASY': ['GRASS', 'WATER', 'STONE', 'TREE', 'WOOD', 'PATH', 'COAL', 'IRON', 'DIAMOND', 'CRAFTING_TABLE', 'FURNACE', 'PLANT'],
        'MEDIUM': ['INVALID', 'OUT_OF_BOUNDS', 'GRASS', 'WATER', 'STONE', 'TREE', 'WOOD', 'PATH', 'COAL', 'IRON', 'DIAMOND', 'CRAFTING_TABLE', 'FURNACE', 'SAND', 'LAVA', 'PLANT', 'RIPE_PLANT', 'WALL', 'DARKNESS', 'WALL_MOSS', 'STALAGMITE', 'SAPPHIRE', 'RUBY', 'CHEST', 'FOUNTAIN', 'FIRE_GRASS', 'ICE_GRASS', 'GRAVEL', 'FIRE_TREE', 'ICE_SHRUB', 'ENCHANTMENT_TABLE_FIRE', 'ENCHANTMENT_TABLE_ICE']
    }
    sides = ['RIGHT', 'LEFT', 'TOP', 'BOTTOM']
    distances = list(range(1, 6))
    
    combinations = set()
    
    while len(combinations) < num_goals:
        object_name = random.choice(place_objects[difficulty])
        target_name = random.choice(target_objects[difficulty])
        side = random.choice(sides)
        distance = random.choice(distances)
        
        goal = f"[{difficulty}] PLACE {object_name} {distance} BLOCKS TO THE {side} OF {target_name}"
        combinations.add(goal)
    
    return list(combinations)