from craftext.checkers_jax.relevant import place_object_relevant_to
from craftext.scenarios.constants import BlockType

easy_test_parafrased = {
    "place_test_parafrased_easy_1": {
        'instruction': "Position a crafting table just one tile to the right of the water source.",
        'instruction_paraphrases': [
            "Set a crafting table one tile to the right of the water.",
            "Place a crafting table exactly one block away from the water, on its right.",
            "Put a crafting table one unit to the right of the water.",
            "Arrange a crafting table one step away from the water, on the right side.",
            "Position a crafting table just one tile away from the water, on the right."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.CRAFTING_TABLE, BlockType.WATER, 0, 1
        ),
        'complexity': "easy-pease"
    },

    "place_test_parafrased_easy_2": {
        'instruction': "Arrange a furnace two blocks directly below the stone structure.",
        'instruction_paraphrases': [
            "Put a furnace two tiles below the stone block.",
            "Place a furnace exactly two blocks under the stone.",
            "Set a furnace two units away from the stone, below it.",
            "Arrange a furnace two steps south of the stone block.",
            "Position a furnace two tiles away from the stone, downwards."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.FURNACE, BlockType.STONE, 3, 2
        ),
        'complexity': "easy-pease"
    },

    "place_test_parafrased_easy_3": {
        'instruction': "Put the crafting table one unit on top of the tree.",
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
    }
}

easy_test_other_paramets = {
    "place_easy_test_other_paramets_easy_4": {
        'instruction': "Place a furnace 1 step to the left of the lake.",
        'instruction_paraphrases': [
            "Set a furnace one tile left of the lake.",
            "Place a furnace exactly one space to the left of the lake.",
            "Position a furnace block one unit to the left of the lake.",
            "Arrange a furnace one step left of the lake.",
            "Put a furnace block one tile away from the lake, on the left."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.FURNACE, BlockType.WATER, 1, 1
        ),
        'complexity': "easy"
    },

    "place_easy_test_other_paramets_easy_5": {
        'instruction': "Place a crafting table 3 steps above the coal block.",
        'instruction_paraphrases': [
            "Put a crafting table three tiles above the coal block.",
            "Position a crafting table exactly three spaces above the coal.",
            "Set a crafting table three units away from the coal, upwards.",
            "Arrange a crafting table three steps north of the coal block.",
            "Place a crafting table three tiles away from the coal, above it."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.CRAFTING_TABLE, BlockType.COAL, 2, 3
        ),
        'complexity': "medium"
    },

    "place_easy_test_other_paramets_easy_6": {
        'instruction': "Put a furnace 2 steps to the right of the tree.",
        'instruction_paraphrases': [
            "Place a furnace two tiles to the right of the tree.",
            "Position a furnace exactly two spaces right of the tree.",
            "Set a furnace two units away from the tree, on its right.",
            "Arrange a furnace two steps to the right of the tree.",
            "Put a furnace block two tiles away from the tree, to the right."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, BlockType.FURNACE, BlockType.TREE, 0, 2
        ),
        'complexity': "medium"
    }
}
