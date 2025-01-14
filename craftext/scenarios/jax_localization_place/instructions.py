from craftext.checkers_jax.relevant import place_object_relevant_to
from craftext.scenarios.constants import BlockType


one = {"one_1": {
        'instruction': "Put a crafting table 1 step above the tree.",
        'instruction_paraphrases': [
            "Place a crafting table one tile above the tree.",
            "Position a crafting table exactly one block up from the tree.",
            "Set up a crafting table one unit away from the tree, above it.",
            "Arrange a crafting table one step above the tree.",
            "Put a crafting table one tile away from the tree, on top."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.CRAFTING_TABLE, BlockType.TREE, 2, 1
        ),
        'complexity': "easy-pease"
    },
       "one_2": {
        'instruction': "Put a crafting table 1 step above the tree.",
        'instruction_paraphrases': [
            "Place a crafting table one tile above the tree.",
            "Position a crafting table exactly one block up from the tree.",
            "Set up a crafting table one unit away from the tree, above it.",
            "Arrange a crafting table one step above the tree.",
            "Put a crafting table one tile away from the tree, on top."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.CRAFTING_TABLE, BlockType.TREE, 2, 1
        ),
        'complexity': "easy-pease"
    },
      }
easy = {}

medium = {}
from craftext.scenarios.parce_dataset import update_previous_dict

easy = update_previous_dict(easy, "../craftext/scenarios/jax_localization_place/instructions/train/easy", "localization_place")
medium = update_previous_dict(medium, "../craftext/scenarios/jax_localization_place/instructions/train/medium", "localization_place")
