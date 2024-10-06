from craftext.checkers_jax.relevant import place_object_relevant_to
easy = {
    "small_train_easy_peasy_1": {
        'instruction': "Place a crafting table next to the water, 1 step away to the right.",
        'instruction_paraphrases': [
            "Set a crafting table one tile to the right of the water.",
            "Place a crafting table exactly one block away from the water, on its right.",
            "Put a crafting table one unit to the right of the water.",
            "Arrange a crafting table one step away from the water, on the right side.",
            "Position a crafting table just one tile away from the water, on the right."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, 11, 3, 0, 1  # CRAFTING_TABLE = 11, WATER = 3
        ),
        'complexity': "easy-pease"
    },

    "small_train_easy_peasy_2": {
        'instruction': "Place a furnace below the stone, 2 steps away.",
        'instruction_paraphrases': [
            "Put a furnace two tiles below the stone block.",
            "Place a furnace exactly two blocks under the stone.",
            "Set a furnace two units away from the stone, below it.",
            "Arrange a furnace two steps south of the stone block.",
            "Position a furnace two tiles away from the stone, downwards."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, 12, 4, 3, 2  # FURNACE = 12, STONE = 4
        ),
        'complexity': "easy-pease"
    },

    "small_train_easy_peasy_3": {
        'instruction': "Put a crafting table 1 step above the tree.",
        'instruction_paraphrases': [
            "Place a crafting table one tile above the tree.",
            "Position a crafting table exactly one block up from the tree.",
            "Set up a crafting table one unit away from the tree, above it.",
            "Arrange a crafting table one step above the tree.",
            "Put a crafting table one tile away from the tree, on top."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, 11, 5, 2, 1  # CRAFTING_TABLE = 11, TREE = 5
        ),
        'complexity': "easy-pease"
    },

    "small_train_easy_1": {
        'instruction': "Place a furnace 3 steps to the left of the path.",
        'instruction_paraphrases': [
            "Set a furnace three tiles left of the path.",
            "Place a furnace exactly three spaces to the left of the path.",
            "Position a furnace block three units to the left of the path.",
            "Arrange a furnace three steps left of the path.",
            "Put a furnace block three tiles away from the path, on the left."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, 12, 7, 1, 3  # FURNACE = 12, PATH = 7
        ),
        'complexity': "easy"
    },

    "small_train_medium_1": {
        'instruction': "Place a crafting table 2 steps above the coal block.",
        'instruction_paraphrases': [
            "Put a crafting table two tiles above the coal block.",
            "Position a crafting table exactly two spaces above the coal.",
            "Set a crafting table two units away from the coal, upwards.",
            "Arrange a crafting table two steps north of the coal block.",
            "Place a crafting table two tiles away from the coal, above it."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, 11, 8, 2, 2  # CRAFTING_TABLE = 11, COAL = 8
        ),
        'complexity': "medium"
    },

    "small_train_medium_2": {
        'instruction': "Put a furnace 1 step to the right of the iron block.",
        'instruction_paraphrases': [
            "Place a furnace one tile to the right of the iron block.",
            "Position a furnace exactly one space right of the iron block.",
            "Set a furnace one unit away from the iron, on its right.",
            "Arrange a furnace one step to the right of the iron block.",
            "Put a furnace block one tile away from the iron block, to the right."
        ],
        'check_lambda': lambda game_data, ix: place_object_relevant_to(
            game_data, 12, 9, 0, 1  # FURNACE = 12, IRON = 9
        ),
        'complexity': "medium"
    }
}

complex = {
    "task_easy_pease_1": {
        'instruction': "Place a crafting table to the right of the plant, 2 steps away.",
        'instruction_paraphrases': [
            "Set up a crafting table two tiles to the right of the plant.",
            "Put a crafting table exactly two blocks away from the plant, on the right.",
            "Position a crafting table 2 spaces to the right of the plant.",
            "Place a crafting table to the right of the plant, 2 units away.",
            "Arrange a crafting table two steps to the right of the plant."
        ],
        'check_lambda': lambda game_data: place_object_relevant_to(
            game_data, 11, 15, 0, 2  # CRAFTING_TABLE = 11, PLANT = 15
        ),
        'complexity': "easy-pease"
    },

    "task_easy_pease_2": {
        'instruction': "Put a plant below the crafting table, 1 step away.",
        'instruction_paraphrases': [
            "Place a plant one tile beneath the crafting table.",
            "Position a plant exactly one step below the crafting table.",
            "Set a plant block just one unit down from the crafting table.",
            "Put a plant block one tile away from the crafting table, below it.",
            "Arrange a plant block right under the crafting table, 1 step away."
        ],
        'check_lambda': lambda game_data: place_object_relevant_to(
            game_data, 15, 11, 3, 1  # PLANT = 15, CRAFTING_TABLE = 11
        ),
        'complexity': "easy-pease"
    },

    "task_easy_1": {
        'instruction': "Place a furnace to the left of the stone block, 3 steps away.",
        'instruction_paraphrases': [
            "Set a furnace three tiles to the left of the stone.",
            "Put a furnace exactly three spaces left of the stone block.",
            "Position a furnace block three units to the left of the stone.",
            "Place a furnace to the left of the stone block, 3 tiles away.",
            "Arrange a furnace three steps to the left of the stone."
        ],
        'check_lambda': lambda game_data: place_object_relevant_to(
            game_data, 12, 4, 1, 3  # FURNACE = 12, STONE = 4
        ),
        'complexity': "easy"
    },

    "task_easy_2": {
        'instruction': "Place a stone to the right of the furnace, 2 steps away.",
        'instruction_paraphrases': [
            "Put a stone block two tiles to the right of the furnace.",
            "Position a stone exactly two blocks to the right of the furnace.",
            "Set a stone two units away from the furnace, on the right.",
            "Place a stone block two steps to the right of the furnace.",
            "Arrange a stone block two tiles away from the furnace on its right."
        ],
        'check_lambda': lambda game_data: place_object_relevant_to(
            game_data, 4, 12, 0, 2  # STONE = 4, FURNACE = 12
        ),
        'complexity': "easy"
    },

    "task_hard_1": {
        'instruction': "Position a chest above the fountain, 4 steps away.",
        'instruction_paraphrases': [
            "Set a chest four tiles above the fountain.",
            "Place a chest block exactly four blocks up from the fountain.",
            "Position a chest four units away from the fountain, above it.",
            "Arrange a chest four tiles north of the fountain.",
            "Place a chest block four steps up from the fountain."
        ],
        'check_lambda': lambda game_data: place_object_relevant_to(
            game_data, 23, 24, 2, 4  # CHEST = 23, FOUNTAIN = 24
        ),
        'complexity': "hard"
    },

    "task_hard_2": {
        'instruction': "Place an enchantment table (fire) to the right of the plant, 5 steps away.",
        'instruction_paraphrases': [
            "Put an enchantment table (fire) five tiles to the right of the plant.",
            "Position an enchantment table of fire type exactly five blocks to the right of the plant.",
            "Set an enchantment table (fire) five units away from the plant, on the right.",
            "Place an enchantment table (fire) five steps to the right of the plant.",
            "Arrange an enchantment table (fire) five tiles away from the plant on its right."
        ],
        'check_lambda': lambda game_data: place_object_relevant_to(
            game_data, 30, 15, 0, 5  # ENCHANTMENT_TABLE_FIRE = 30, PLANT = 15
        ),
        'complexity': "hard"
    }
}
