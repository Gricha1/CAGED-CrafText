from craftext.checkers_jax.building import is_square_formed
from craftext.scenarios.constants import BlockType
from craftext.scenarios.parce_dataset import update_previous_dict

one = {
    "squere_one": {
        'check_lambda': lambda game_data, ix: is_square_formed(game_data, ix, BlockType.STONE, 2),
        'str_check_lambda': "is_square_formed(game_data, ix, BlockType.STONE, 2)",
        'instruction': "Build a 2x2 square of stone blocks.",
        'instruction_paraphrases': [
            "Form a sturdy stone square, 2 blocks long on each side.",
            "Construct a small plaza of stones, arranged in a 2 by 2 block formation.",
            "Lay out stone blocks to create a strong 2x2 enclosed area.",
            "Place stones to build a firm square structure, with 2 blocks making up each side.",
            "Create a 2x2 stone foundation, ensuring that each corner of the square is defined."
        ]
    }, 
    "squere_one_1": {
        'check_lambda': lambda game_data, ix: is_square_formed(game_data,ix, BlockType.STONE, 2),
        'str_check_lambda': "is_square_formed(game_data, ix, BlockType.STONE, 2)",
        'instruction': "Build a 2x2 square of stone blocks.",
        'instruction_paraphrases': [
            "Form a sturdy stone square, 2 blocks long on each side.",
            "Construct a small plaza of stones, arranged in a 2 by 2 block formation.",
            "Lay out stone blocks to create a strong 2x2 enclosed area.",
            "Place stones to build a firm square structure, with 2 blocks making up each side.",
            "Create a 2x2 stone foundation, ensuring that each corner of the square is defined."
        ]
    }
}
easy = {}
        
        
medium = {}


from craftext.scenarios.constants import base_path
import os

easy = update_previous_dict(
    easy, 
    os.path.join(base_path, "jax_build_squere/instructions/train/easy"), 
    "build_squere"
)

medium = update_previous_dict(
    medium, 
    os.path.join(base_path, "jax_build_squere/instructions/train/medium"), 
    "build_squere"
)