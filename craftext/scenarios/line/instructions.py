from craftext.checkers.scenarios_building import is_line_formed

instructions = {
    "57": {
        'instruction': "Align three crafting tables in a straight horizontal line.",
        'instruction_paraphrases': [
            "Place three crafting tables in a single horizontal row.",
            "Arrange three crafting tables side by side in a straight line.",
            "Set up three crafting tables in a horizontal line, each one next to the other.",
            "Create a straight horizontal sequence of three crafting tables.",
            "Position three crafting tables to form a continuous horizontal line."
        ],
        'check_lambda': lambda game_data: is_line_formed(game_data, "CRAFTING_TABLE", 3, check_diagonal=False)
    },
    "58": {
        'instruction': "Position four furnaces in a vertical line.",
        'instruction_paraphrases': [
            "Arrange four furnaces in a single vertical column.",
            "Place four furnaces one on top of the other in a vertical line.",
            "Set up four furnaces in a straight vertical line.",
            "Create a vertical alignment of four furnaces, stacking them in a column.",
            "Form a vertical line by placing four furnaces one above the other."
        ],
        'check_lambda': lambda game_data: is_line_formed(game_data, "FURNACE", 4, check_diagonal=False)
    },
    "59": {
        'instruction': "Arrange five chests in a diagonal line, from top-left to bottom-right.",
        'instruction_paraphrases': [
            "Position five chests diagonally from the top-left corner to the bottom-right.",
            "Create a diagonal line of five chests starting from the top-left.",
            "Set up five chests in a diagonal pattern from the top-left to the bottom-right.",
            "Form a diagonal sequence with five chests, extending from the upper left to the lower right.",
            "Align five chests in a diagonal line, running from the top-left to the bottom-right."
        ],
        'check_lambda': lambda game_data: is_line_formed(game_data, "CHEST", 5, check_diagonal=True)
    },
    "60": {
        'instruction': "Place six fountains in a continuous line, either horizontally or vertically.",
        'instruction_paraphrases': [
            "Arrange six fountains in a straight line, either in a row or a column.",
            "Form a line of six fountains, oriented either horizontally or vertically.",
            "Position six fountains in a single straight line, regardless of direction.",
            "Set up a straight line of six fountains, aligned either horizontally or vertically.",
            "Create a continuous sequence of six fountains, in either a horizontal or vertical direction."
        ],
        'check_lambda': lambda game_data: is_line_formed(game_data, "FOUNTAIN", 6, check_diagonal=False)
    },
    "61": {
        'instruction': "Design a line of seven plants, ensuring they connect diagonally.",
        'instruction_paraphrases': [
            "Align seven plants in a diagonal line, making sure each plant is connected.",
            "Set up seven plants in a continuous diagonal line.",
            "Create a diagonal pattern with seven plants, ensuring they form a straight line.",
            "Arrange seven plants in a diagonal sequence, each one connected to the next.",
            "Position seven plants in a straight diagonal line, making sure they are aligned."
        ],
        'check_lambda': lambda game_data: is_line_formed(game_data, "PLANT", 7, check_diagonal=True)
    }
}
