from craftext.checkers_jax.building import is_square_formed
from craftext.scenarios.constants import BlockType

easy_test_parafrased = {
    "squere_parafrased_1": {
        'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, BlockType.CRAFTING_TABLE, 2),
        'instruction_paraphrases': [
            "Build a small crafting table structure in a 2x2 layout.",
            "Arrange four crafting tables into a square with 2 units per side.",
            "Form a 2 by 2 grid using crafting tables to create a crafting area.",
            "Set up crafting tables to form a compact 2x2 shape.",
            "Create a crafting block by placing 4 tables in a square."
        ]
    },
    "squere_parafrased_2": {
        'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix,BlockType.STONE, 2),
        'instruction_paraphrases': [
            "Arrange stones to create a sturdy 2x2 formation.",
            "Build a small square of stones, with each side made of 2 blocks.",
            "Place 4 stone blocks in a neat 2 by 2 pattern.",
            "Create a stone area by arranging blocks in a 2x2 grid.",
            "Form a compact square of stones using 4 blocks."
        ]
    },

    "squere_parafrased_3": {
        'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, BlockType.STONE, 3),
        'instruction_paraphrases': [
            "Create a large stone square, 3 units long on each side.",
            "Form a strong 3x3 structure using stone blocks.",
            "Build a wide 3 by 3 grid using stones to create a solid foundation.",
            "Arrange stones in a 3x3 formation, forming a large square.",
            "Set up a stone square where each side consists of 3 blocks."
        ]
    },
    "squere_parafrased_4": {
        'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, BlockType.PLANT, 3),
        'instruction_paraphrases': [
            "Arrange plants to create a 3x3 square garden area.",
            "Build a large green patch by placing plants in a 3 by 3 pattern.",
            "Form a garden area using plants, arranged in a 3x3 grid.",
            "Create a 3x3 plant formation to complete the garden.",
            "Set up a large square of plants, with each side consisting of 3 plants."
        ]
    }
}


easy_test_other_paramets = {
    "squere_other_1": {
        'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, BlockType.CRAFTING_TABLE, 3),
        'instruction_paraphrases': [
            "Build a 3x3 square of crafting tables.",
            "Create a larger crafting area by arranging crafting tables in a 3x3 shape.",
            "Form a 3x3 grid using crafting tables for an extended workspace.",
            "Set up a big crafting table block, arranging them into a 3 by 3 formation.",
            "Construct a 3x3 square with crafting tables, expanding the crafting zone."
        ]
    },
    
        "squere_other_2": {
        'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, BlockType.FURNACE, 2),
        'instruction_paraphrases': [
            "Construct a furnace block by arranging 4 furnaces in a 2x2 shape.",
            "Build a small square using furnaces, with 2 units per side.",
            "Create a furnace layout in a neat 2 by 2 formation.",
            "Set up furnaces to form a 2x2 block, creating a heating area.",
            "Form a square of furnaces, with each side consisting of 2 furnaces."
        ]
    },
    
    "squere_other_3": {
        'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, BlockType.PLANT, 4),
        'instruction_paraphrases': [
            "Build a 4x4 green patch by arranging plants in a grid.",
            "Form a large 4 by 4 garden area using plants.",
            "Create a 4x4 grid of plants for a spacious garden.",
            "Arrange plants in a 4x4 layout to create a large green zone.",
            "Set up a 4x4 square of plants to grow a bigger garden patch."
        ]
    },
    
        "squere_other_4": {
        'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, BlockType.PLANT, 2),
        'instruction_paraphrases': [
            "Build a green zone by arranging plants in a 2x2 layout.",
            "Set up a square of plants, with each side measuring 2 units.",
            "Form a compact garden area using plants arranged in a 2 by 2 pattern.",
            "Create a plant square, placing 2 plants on each side.",
            "Arrange 4 plants in a tight 2x2 grid to form a small garden patch."
        ]
    }
}
