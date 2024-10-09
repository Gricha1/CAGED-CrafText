from craftext.checkers_jax.building import is_line_formed, is_square_formed
from craftext.scenarios.constants import BlockType

easy_test_parafrased = {
    'line_easy_1': {
        'instruction': "Make a line of 2 blocks using table.",
        'instruction_paraphrases': [
            "Construct a row of 2 elements with the crafting station.",
            "Place 2 blocks in a straight path using the workbench.",
            "Use the crafting table to arrange a line of 2 blocks.",
            "Form a sequence of 2 items with the crafting platform.",
            "Set up a line of 2 pieces using the crafting table."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.CRAFTING_TABLE, 2, check_diagonal=False)
    },
    'line_easy_2': {
        'instruction': "Make a diagonal line of 3 blocks using table.",
        'instruction_paraphrases': [
            "Create a diagonal arrangement of 3 units with the crafting bench.",
            "Set up 3 blocks diagonally using the workbench.",
            "Use the crafting station to arrange 3 blocks in a diagonal formation.",
            "Form a slanted row of 3 items with the crafting table.",
            "Arrange 3 blocks in a diagonal line using the crafting table."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.CRAFTING_TABLE, 3, check_diagonal=True)
    },
    'line_easy_4': {
        'instruction': "Make a line of 2 blocks using furnace.",
        'instruction_paraphrases': [
            "Create a row of 2 blocks with the furnace.",
            "Arrange 2 blocks in a line using the smelter.",
            "Use the furnace to position 2 blocks in a straight line.",
            "Place 2 items in a sequence with the furnace.",
            "Form a straight row of 2 blocks using the heating station."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix,BlockType.FURNACE, 2, check_diagonal=False)
    }
}

easy_test_other_paramets = {
    'line_medium_1': {
        'instruction': "Make a vertical line of 4 blocks using stone.",
        'instruction_paraphrases': [
            "Set up a vertical row of 4 stone blocks.",
            "Place 4 stones in a straight vertical line.",
            "Use stone to arrange 4 blocks in a vertical column.",
            "Create a vertical line of 4 stone blocks.",
            "Arrange 4 stone pieces in a vertical formation."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix,BlockType.STONE, 4, check_diagonal=False)
    },
    'line_medium_2': {
        'instruction': "Make a square of 4 blocks using crafting table.",
        'instruction_paraphrases': [
            "Construct a square formation of 4 blocks with the crafting table.",
            "Arrange 4 blocks in a square using the crafting station.",
            "Use the crafting table to form a square of 4 units.",
            "Place 4 items in a square shape with the crafting table.",
            "Set up a square of 4 blocks using the crafting platform."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.CRAFTING_TABLE, 4)
    },
    'line_medium_3': {
        'instruction': "Make a horizontal line of 5 blocks using furnace.",
        'instruction_paraphrases': [
            "Create a horizontal line of 5 blocks with the furnace.",
            "Arrange 5 blocks in a straight horizontal line using the furnace.",
            "Use the furnace to place 5 blocks in a row.",
            "Form a sequence of 5 items horizontally using the smelter.",
            "Set up a straight row of 5 blocks with the furnace."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.FURNACE, 5, check_diagonal=False)
    }
}

