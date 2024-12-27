import random

def generate_example_goals(num_goals: int, difficulty="EASY") -> list[str]:
    """
    Generate a list of unique text-based goals for conditional placing with difficulty levels.
    
    Args:
        num_goals (int): Number of goals to generate.
    
    Returns:
        list[str]: List of unique conditional placing goals.
    """
    inventory_items = {
        'EASY': ['WOOD', 'STONE', 'COAL', 'IRON', 'DIAMOND', 'SAPLING', 'WOOD_PICKAXE', 'STONE_PICKAXE', 'IRON_PICKAXE', 'WOOD_SWORD', 'STONE_SWORD', 'IRON_SWORD'],
        'MEDIUM': ['WOOD', 'STONE', 'COAL', 'IRON', 'DIAMOND', 'SAPLING', 'WOOD_PICKAXE', 'STONE_PICKAXE', 'IRON_PICKAXE', 'WOOD_SWORD', 'STONE_SWORD', 'IRON_SWORD']
    }
    placeable_objects = {
        'EASY': ['STONE', 'CRAFTING_TABLE', 'PLANT', 'FURNACE'],
        'MEDIUM': ['STONE', 'CRAFTING_TABLE', 'PLANT', 'FURNACE', 'ENCHANTMENT_TABLE_FIRE', 'ENCHANTMENT_TABLE_ICE']
    }
    counts_to_collect = list(range(1, 6))  # Collect between 1 and 5 items
    counts_to_stand = list(range(1, 6))   # Place between 1 and 5 items
    
    combinations = set()
    
    while len(combinations) < num_goals:
        inventory_item = random.choice(inventory_items[difficulty])
        object_to_place = random.choice(placeable_objects[difficulty])
        count_collect = random.choice(counts_to_collect)
        count_stand = random.choice(counts_to_stand)
        
        goal = (f"[{difficulty}] COLLECT {count_collect} {inventory_item} AND PLACE {count_stand} {object_to_place}")
        combinations.add(goal)
    
    return list(combinations)
    