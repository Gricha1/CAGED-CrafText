from craftext.checkers_jax.building import is_line_formed
from craftext.scenarios.constants import BlockType
from craftext.scenarios.parce_dataset import update_previous_dict
one = {
    'line_one_1': {
        'instruction': "Make a line of 2 blocks using table.",
        'instruction_paraphrases': [
            "Construct a row of 2 pieces with the crafting station.",
            "Place 2 units in a straight row using the workbench.",
            "Use the crafting table to form a row of 2 items.",
            "Arrange a sequence of 2 blocks with the crafting platform.",
            "Create a straight formation of 2 blocks with the crafting table."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.CRAFTING_TABLE, 2, check_diagonal=False),
        'str_check_lambda': "is_line_formed(game_data,ix, BlockType.CRAFTING_TABLE, 2, check_diagonal=False)"
    },
    
    'line_one_2': {
        'instruction': "Make a line of 2 blocks using table.",
        'instruction_paraphrases': [
            "Construct a row of 2 pieces with the crafting station.",
            "Place 2 units in a straight row using the workbench.",
            "Use the crafting table to form a row of 2 items.",
            "Arrange a sequence of 2 blocks with the crafting platform.",
            "Create a straight formation of 2 blocks with the crafting table."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.CRAFTING_TABLE, 2, check_diagonal=False),
        'str_check_lambda': "is_line_formed(game_data,ix, BlockType.CRAFTING_TABLE, 2, check_diagonal=False)"
    },
}
easy = {
    'line_easy_1': {
        'instruction': "Make a line of 2 blocks using table.",
        'instruction_paraphrases': [
            "Construct a row of 2 pieces with the crafting station.",
            "Place 2 units in a straight row using the workbench.",
            "Use the crafting table to form a row of 2 items.",
            "Arrange a sequence of 2 blocks with the crafting platform.",
            "Create a straight formation of 2 blocks with the crafting table."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 2, check_diagonal=False),
        'str_check_lambda': "is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 2, check_diagonal=False)"
    },
    'line_easy_2': {
        'instruction': "Make a diagonal line of 3 blocks using table.",
        'instruction_paraphrases': [
            "Construct a slanted row of 3 items with the crafting station.",
            "Place 3 blocks in a diagonal sequence using the workbench.",
            "Use the crafting platform to arrange a diagonal row of 3 pieces.",
            "Arrange a sloped line of 3 units using the crafting bench.",
            "Create a diagonal sequence of 3 items with the crafting table."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 3, check_diagonal=True),
        'str_check_lambda': "is_line_formed(game_data, ix, BlockType.CRAFTING_TABLE, 3, check_diagonal=True)"
    },
    'line_easy_3': {
        'instruction': "Make a diagonal line of 2 blocks using stone.",
        'instruction_paraphrases': [
            "Construct a slanted row of 2 stones.",
            "Place 2 stone blocks in a diagonal sequence.",
            "Use stones to arrange a diagonal line of 2 pieces.",
            "Arrange a sloped row of 2 stone units.",
            "Create a diagonal sequence of 2 stone blocks."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data, ix, BlockType.STONE, 2, check_diagonal=True),
        'str_check_lambda': "is_line_formed(game_data, ix, BlockType.STONE, 2, check_diagonal=True)"
    },
    'line_easy_4': {
        'instruction': "Make a line of 2 blocks using furnace.",
        'instruction_paraphrases': [
            "Construct a straight row of 2 units with the furnace.",
            "Place 2 blocks in a line using the heating station.",
            "Use the furnace to form a sequence of 2 items.",
            "Arrange a straight line of 2 pieces with the smelter.",
            "Create a row of 2 blocks using the furnace."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data, ix, BlockType.FURNACE, 2, check_diagonal=False),
        'str_check_lambda': "is_line_formed(game_data, ix, BlockType.FURNACE, 2, check_diagonal=False)"
    },
    'line_easy_5': {
        'instruction': "Make a horizontal line of 4 blocks using stone.",
        'instruction_paraphrases': [
            "Construct a straight line of 4 stone blocks.",
            "Place 4 stone units in a horizontal row.",
            "Use stones to form a line of 4 blocks in a straight path.",
            "Arrange 4 stones in a straight sequence.",
            "Create a horizontal formation of 4 stone blocks."
        ],
        'check_lambda': lambda game_data, ix: is_line_formed(game_data, ix, BlockType.STONE, 4, check_diagonal=False),
        'str_check_lambda': "is_line_formed(game_data, ix, BlockType.STONE, 4, check_diagonal=False)"
    }
}


medium = {}

from craftext.scenarios.constants import base_path
import os

easy = update_previous_dict(
    easy, 
    os.path.join(base_path, "jax_build_line/instructions/train/easy"), 
    "build_line"
)

medium = update_previous_dict(
    medium, 
    os.path.join(base_path, "jax_build_line/instructions/train/medium"), 
    "build_line"
)