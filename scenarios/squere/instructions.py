from text_craftax_base_checkers.scenarios_building import is_cross_formed

instructions = {
    "instruction_001": {
        'instruction': "Arrange four crafting tables into a compact square shape.",
        'instruction_paraphrases': [
            "Place four crafting tables in the form of a small square.",
            "Organize the crafting tables into a tight square formation.",
            "Create a square by positioning four crafting tables next to each other.",
            "Form a neat square using four crafting tables arranged side by side.",
            "Establish a square layout by placing four crafting tables in a symmetrical pattern."
        ],
        'check_lambda': lambda game_data: is_square_formed(game_data, "CRAFTING_TABLE", 2)
    },
    "instruction_002": {
        'instruction': "Create a square grid using nine furnaces, ensuring each furnace connects to its neighbor.",
        'instruction_paraphrases': [
            "Set up a 3x3 square by placing nine furnaces in a grid.",
            "Arrange the furnaces into a connected square formation with three on each side.",
            "Construct a square grid using nine furnaces, aligned in three rows and three columns.",
            "Form a square pattern by placing nine furnaces so that each connects to another.",
            "Design a square-shaped grid with nine furnaces, ensuring they're all adjacent."
        ],
        'check_lambda': lambda game_data: is_square_formed(game_data, "FURNACE", 3)
    },
    "instruction_003": {
        'instruction': "Organize sixteen chests into four rows and four columns, forming a square.",
        'instruction_paraphrases': [
            "Place sixteen chests in a 4x4 square arrangement.",
            "Form a larger square by arranging sixteen chests into four lines of four.",
            "Create a square pattern with sixteen chests, positioning them in rows and columns.",
            "Construct a square grid using sixteen chests, with four in each row and column.",
            "Design a square formation by setting up sixteen chests in a grid pattern."
        ],
        'check_lambda': lambda game_data: is_square_formed(game_data, "CHEST", 4)
    },
    "instruction_004": {
        'instruction': "Use twenty-five fountains to create a large square pattern.",
        'instruction_paraphrases': [
            "Arrange twenty-five fountains into a big square.",
            "Construct a spacious square grid with twenty-five fountains, ensuring they're evenly spaced.",
            "Form a square shape using twenty-five fountains, organized into five rows and five columns.",
            "Create a sizable square by placing twenty-five fountains in a grid formation.",
            "Design a large square layout with twenty-five fountains, connecting them in a 5x5 pattern."
        ],
        'check_lambda': lambda game_data: is_square_formed(game_data, "FOUNTAIN", 5)
    },
    "instruction_005": {
        'instruction': "Establish a pattern where thirty-six plants form a balanced square.",
        'instruction_paraphrases': [
            "Place thirty-six plants in a symmetrical square formation.",
            "Arrange thirty-six plants to form a square pattern, with each plant aligned with its neighbors.",
            "Create a square using thirty-six plants, organized into a structured grid.",
            "Design a balanced square layout with thirty-six plants, ensuring equal spacing.",
            "Form a large square by positioning thirty-six plants into six rows and six columns."
        ],
        'check_lambda': lambda game_data: is_square_formed(game_data, "PLANT", 6)
    }
}
