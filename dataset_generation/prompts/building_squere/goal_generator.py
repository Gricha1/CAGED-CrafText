import random
def generate_example_goals(num_goals: int, difficulty="EASY") -> list[str]:
    """
    Generate a list of unique text-based goals describing block placement patterns.
    
    Args:
        num_goals (int): Number of goals to generate.
    
    Returns:
        list[str]: List of unique goal descriptions.
    """
    block_types = {
        'EASY': ['STONE', 'CRAFTING_TABLE', 'PLANT', 'FURNACE'],
        'MEDIUM': ['STONE', 'CRAFTING_TABLE', 'PLANT', 'FURNACE', 'ENCHANTMENT_TABLE_FIRE', 'ENCHANTMENT_TABLE_ICE']
    }
    
    if difficulty=='EASY':
        sizes = list(range(2, 5))
    else:
        sizes = list(range(2, 8))
    
    combinations = set()
    
    while len(combinations) < num_goals:
       # difficulty = random.choice(list(block_types.keys()))
        block = random.choice(block_types[difficulty])
        size = random.choice(sizes)
        
        goal = f"SQUERE OF {block} WITH SIDE SIZE {size}"
        combinations.add(goal)
    
    return list(combinations)