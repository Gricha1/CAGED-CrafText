from craftext.enviroment.scenarious.checkers.target_state import TargetState, BuildLineState
from craftext.enviroment.craftext_constants import Scenarios, BlockType


# one = {
#     'line_one_1': {
#         'instruction': "Make a line of 2 blocks using table.",
#         'instruction_paraphrases': [
#             "Construct a row of 2 pieces with the crafting station.",
#             "Place 2 units in a straight row using the workbench.",
#             "Use the crafting table to form a row of 2 items.",
#             "Arrange a sequence of 2 blocks with the crafting platform.",
#             create a straight formation of 2 blocks with the crafting table."
#         ],
#         'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.CRAFTING_TABLE, 2, is_diagonal=False),
#         'str_check_lambda': "is_line_formed(game_data,ix, BlockType.CRAFTING_TABLE, 2, is_diagonal=False)"
#     },
    
#     'line_one_2': {
#         'instruction': "Make a line of 2 blocks using table.",
#         'instruction_paraphrases': [
#             "Construct a row of 2 pieces with the crafting station.",
#             "Place 2 units in a straight row using the workbench.",
#             "Use the crafting table to form a row of 2 items.",
#             "Arrange a sequence of 2 blocks with the crafting platform.",
#             create a straight formation of 2 blocks with the crafting table."
#         ],
#         'check_lambda': lambda game_data, ix: is_line_formed(game_data,ix, BlockType.CRAFTING_TABLE, 2, is_diagonal=False),
#         'str_check_lambda': "is_line_formed(game_data,ix, BlockType.CRAFTING_TABLE, 2, is_diagonal=False)"
#     },
# }

def create_target_state(block_type:int, size:int, is_diagonal:bool):
    target_achievements = BuildLineState(block_type=block_type, size=size, is_diagonal=is_diagonal, radius=10)
    return TargetState(building_line=target_achievements)

easy = {
    "INSTRUCTION_CRAFTING_TABLE_3": {
        "instruction": "Form a square of crafting tables with each side having a length of 3",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "create a square shape by using work benches where the number of work benches on one side is 3",
            "Layout 3 blocks of builder’s table in a square shape.",
            "Make sure to arrange the building blocks in a square shape with each side having 3 of them",
            "Kindly arrange three Crafting platforms on each side to form a square configuration.",
            "I want you to position the construction desks in such a way that they form a square structure with each side containing three desks"
        ],
        "arguments": create_target_state(BlockType.CRAFTING_TABLE, 3, is_diagonal=False),
        "str_check_lambda": "is_square_formed(gd, ix)"
    },
    "INSTRUCTION_FURNACE_6": {
        "instruction": "Verify if there is a square formed of furnace blocks with a side size of 6.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Can you confirm if a square with side-length of 6 is made up of furnace blocks?",
            "Tell me, do we have a furnace blocks square with each side 6 blocks long?",
            "Check for a 6 blocks sided square made entirely out of furnace blocks.",
            "Do assess if we have a square, each side 6 blocks long, made completely from furnace blocks.",
            "Inspect and confirm whether there is a square structure constituted of furnace blocks, six blocks long per side."
        ],
        "arguments": create_target_state(BlockType.FURNACE, 6, is_diagonal=False),
        "str_check_lambda": "is_square_formed(gd, ix)"
    },
    "INSTRUCTION_FURNACE_5": {
        "instruction": "Check if there is a square made of furnace blocks with side length of 5.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Investigate for a square formation of 5x5 using heater blocks.",
            "Look for a quadrant constructed with stove blocks, each side measuring 5 units.",
            "Can you find a geometric square with a side size of 5, constructed from kiln blocks?",
            "Make sure a form of square having dimensions 5 by 5, built using forge blocks is in position?",
            "Inspect for any presence of a geometric configuration resembling a square with side length of 5, created using smelter blocks."
        ],
        "arguments": create_target_state(BlockType.FURNACE, 5, is_diagonal=False),
        "str_check_lambda": "is_square_formed(gd, ix)"
    },
    "INSTRUCTION_CRAFTING_TABLE_7": {
        "instruction": "Check for a distinct square made out of crafting tables with each side having a length of 7 blocks.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Look for a 7x7 crafting block square",
            "Search for a square formation composed of workbenches, with each side consisting of 7 blocks",
            "Confirm if there is a square of crafting tables existing, where each side is equivalent to 7 blocks",
            "Verify the presence of a 7-blocks-wide square of crafting station",
            "Ensure the existence of a perfect square shape made up of 7 blocks per side of crafting tables"
        ],
        "arguments": create_target_state(BlockType.CRAFTING_TABLE, 7, is_diagonal=False),
        "str_check_lambda": "is_square_formed(gd, ix)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_ICE_3": {
        "instruction": "Check for a square formation of Enchantment Table Ice with a side of size 3.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Can you see if there's a square configuration of Ice Enchantment Table each side measuring 3 blocks?",
            "Verify if there's a 3x3 square arrangement of the Ice Magic Desk.",
            "Determine if you have a 9-block square formation of the Frosty Wizard's Stand.",
            "Could you look for a square pattern of Ice Sorcerer's Bench? Each side should have 3 blocks.",
            "Confirm if there exists a square structure of three units on each side of the Cryo Spell Table."
        ],
        "arguments": create_target_state(BlockType.ENCHANTMENT_TABLE_ICE, 3, is_diagonal=False),
        "str_check_lambda": "is_square_formed(gd, ix)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_ICE_size=7": {
        "instruction": "create an enchantment table of ice shaped into a square with each side size 7.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Form a square using an ice enchantment table, each side should be of length 7.",
            "Make a square with sides of 7 units using a table enchanted with ice.",
            "Utilize an enchanted ice table to fabricate a square having side length of seven.",
            "Shape an ice enchantment table into a square with a side length of 7.",
            "With the ice enchantment table, assemble a square where each side measures 7 units."
        ],
        "arguments": create_target_state(BlockType.ENCHANTMENT_TABLE_ICE, 7, is_diagonal=False),
        "str_check_lambda": "is_square_formed(gd, ix)"
    },
    "INSTRUCTION_FURNACE_7": {
        "instruction": "Build a square using furnaces with each side being 7 blocks long.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Craft a square pattern with smelters that has 7 blocks as the dimension.",
            "create a geometric square shape using heating devices where each side is 7 blocks long.",
            "Erect a quadrilateral with furnaces with each of its sides being made of 7 blocks.",
            "Form a furnace square that each edge has a length of 7 blocks.",
            "Construct a four-sided figure using 7 furnaces on each side."
        ],
        "arguments": create_target_state(BlockType.FURNACE, 7, is_diagonal=False),
        "str_check_lambda": "is_square_formed(gd, ix)"
    }
}



medium = {
    "INSTRUCTION_STONE_2": {
        "instruction": "create a diagonal line of stones of size two.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Can you make a diagonal line with two rocks?",
            "Arrange two stones in a diagonal manner",
            "Put two stones in a line, but make sure it's slanted",
            "I want to see two rocks positioned in a slanting line",
            "Could you please arrange a pair of stones diagonally to form a line?"
        ],
        "arguments":create_target_state(BlockType.STONE, 2, is_diagonal=True),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_CRAFTING_TABLE_2": {
        "instruction": "Check if there is a line of Crafting Tables, at least two in size.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Verify if a sequence of at least two Crafting Tables has formed.",
            "Could you see if there's a chain of two or more Crafting Tables?",
            "I need you to ascertain if there is a line up of no less than two Crafting Tables.",
            "Is there a succession of Crafting Tables in at least a duet formation?",
            "Would it be possible to identify a progression of Crafting Tables having a minimum length of two?"
        ],
        "arguments":create_target_state(BlockType.CRAFTING_TABLE, 2, is_diagonal=False),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_FURNACE_3": {
        "instruction": "Form a diagonal line of furnaces of length 3.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Arrange furnaces in a slant line with 3 in total.",
            "Assemble a line of 3 furnaces angled diagonally.",
            "Construct a three-length diagonal row of furnaces.",
            "Diagonally, set up a sequence of three furnaces.",
            "Set about diagonally placing three furnaces in a linear pattern."
        ],
        "arguments":create_target_state(BlockType.FURNACE, 3, is_diagonal=True),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_ICE_2": {
        "instruction": "Form a diagonal line of enchantment ice tables of length 2.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "create a line of two enchantment ice tables slanted diagonally.",
            "Arrange two ice enchantment tables in a sloped arrangement.",
            "Design a diagonal configuration using two ice enchantment tables.",
            "Establish a slanted sequence of two enchantment tables made of ice.",
            "Place two ice enchantment tables in an angular line."
        ],
        "arguments": create_target_state(BlockType.ENCHANTMENT_TABLE_ICE, 2, is_diagonal=True),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_FIRE_7": {
        "instruction": "Check a diagonal line of enchantment table fire blocks of size 7",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Make sure there's a diagonal line of 7 fiery enchantment tables.",
            "Verify the existence of an inclined row consisting of seven blocks of enchantment table on fire.",
            "Could you check if a continuous diagonal sequence of seven fire enchantment tables is present?",
            "I need you to ascertain if a slanting linear arrangement of seven fire enchantment tables exists in the given location.",
            "Confirm whether there exists an unbroken diagonal chain comprising of seven blocks, each comprised of an enchantment table engulfed in flame."
        ],
        "arguments": create_target_state(BlockType.ENCHANTMENT_TABLE_FIRE, 7, is_diagonal=True),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_FURNACE_2": {
        "instruction": "Check if there is a line of two furnaces",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Inspect if there are two furnaces in row",
            "Verify if two furnaces were established in a line",
            "Guarantee that a line is formed with a couple of furnaces",
            "Affirm the existence of a linear arrangement of two furnaces",
            "Ascertain the alignment of pair of furnaces into a straight pattern"
        ],
        "arguments": create_target_state(BlockType.FURNACE, 2, is_diagonal=False),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_PLANT_7": {
        "instruction": "create a diagonal line of plants that is seven units long.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Please, make a seven unit long diagonal line with flora.",
            "Can you construct a slanted line using plants that measures seven units?",
            "I need you to form a diagonal row of vegetation that is seven units in length.",
            "You need to arrange a span of seven plants in a diagonal layout.",
            "Could you design a sequence of vegetation displayed diagonally that extends for seven units?"
        ],
        "arguments": create_target_state(BlockType.PLANT, 7, is_diagonal=True),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_FURNACE_7": {
        "instruction": "create a diagonal line of Furnace with 7 blocks.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Construct a diagonal row of seven Furnaces.",
            "Put together a sequence of Furnace blocks in a bent line, make sure this sequence is 7 blocks long.",
            "Build a series of Furnaces diagonally, and it should consist of 7 Furnaces.",
            "In a diagonal manner, set up a line of 7 Furnace blocks.",
            "Establish a line at an angle using Furnace blocks and ensure this line includes exactly 7 blocks."
        ],
        "arguments": create_target_state(BlockType.FURNACE, 7, is_diagonal=True),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_STONE_2_2": {
        "instruction": "Verify a 2-block line of stone arranged linearly in the game space.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Confirm if there is a linear alignment of stone blocks, each of size two, in the current game area.",
            "Check the game for a straight line configuration of two blocks of rocks.",
            "Establish the existence of a two-piece series of stones positioned in a line within the game environment.",
            "Authenticate if a pair of stone blocks are found in a direct line sequence in the game domain.",
            "Can you substantiate whether a duo of boulders exist in a line formation within the gaming zone?"
        ],
        "arguments": create_target_state(BlockType.STONE, 2, is_diagonal=False),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_FIRE_6": {
        "instruction": "Form a line containing six Enchantment Fire Tables arranged diagonally.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Compose a diagonal sequence incorporating six Enchantment Fire Tables.",
            "Creatively arrange six Enchantment Fire Tables in a straight line at an angle.",
            "Set up a diagonal pattern of six Enchantment Fire Tables.",
            "Organize six Enchantment Fire Tables into a diagonal row.",
            "Construct a diagonal line by using six Fire Enchantment Tables."
        ],
        "arguments": create_target_state(BlockType.ENCHANTMENT_TABLE_FIRE, 6, is_diagonal=True),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_STONE_6": {
        "instruction": "Form a diagonal line of stone blocks with length 6.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Make a diagonal sequence of 6 rock blocks.",
            "I need you to construct a line, on the diagonal axis, using 6 blocks of stone.",
            "create a slanted line using 6 stone blocks.",
            "Place 6 blocks of rock in a diagonal direction, forming a straight line.",
            "Arrange a line with 6 stones diagonally."
        ],
        "arguments": create_target_state(BlockType.STONE, 6, is_diagonal=True),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_ICE_4": {
        "instruction": "Make a diagonal line of enchantment table ice blocks with a size of four.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Can you create a line of enchantment table ice with four blocks diagonally?",
            "Can you place four enchantment table ice blocks in a diagonal arrangement?",
            "Arrange four blocks of enchantment table ice in a diagonal line.",
            "Could you diagonally organize four blocks of enchantment table ice in a row?",
            "Craft a diagonal chain using four pieces of ice enchantment table."
        ],
        "arguments": create_target_state(BlockType.ENCHANTMENT_TABLE_ICE, 4, is_diagonal=True),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_FURNACE_3_2": {
        "instruction": "Form a line of furnace blocks, 3 blocks in length.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Arrange three furnace blocks in a straight line.",
            "Position a series of three furnace blocks linearly.",
            "I need you to line up three furnace blocks, one after the other.",
            "Sequentially place three furnace units in a row.",
            "Erect a horizontally straight sequence of three furnace blocks."
        ],
        "arguments": create_target_state(BlockType.FURNACE, 3, is_diagonal=False),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_PLANT_7_2": {
        "instruction": "create a line of plants with a length of 7 blocks.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Form a sequence of vegetation with a span of seven units.",
            "Make a linear progression of foliage that spans over seven blocks.",
            "Put together a row of greenery extending to seven sections.",
            "Establish a succession of botanical items that provides a stretch of seven blocks.",
            "Set up an arrangement of flora, creating a steady line of seven constituents."
        ],
        "arguments": create_target_state(BlockType.PLANT, 7, is_diagonal=False),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_STONE_4": {
        "instruction": "Form a diagonal line of four rocks.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Draw a line of four stones at an angle.",
            "llace four stones so that they make a diagonal line.",
            "Produce a slanted line using four rocks.",
            "Construct a diagonal arrangement of four stones.",
            "Fabricate a tilted line composed of four rocks."
        ],
        "arguments": create_target_state(BlockType.STONE, 4, is_diagonal=True),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    },
    "INSTRUCTION_CRAFTING_TABLE_7": {
        "instruction": "Check if there is a diagonal line of Crafting Tables of size 7.",
        "scenario_checker": Scenarios.BUILD_LINE,
        "instruction_paraphrases": [
            "Verify the presence of a diagonal sequence of Workbenches of length 7.",
            "Investigate if there exists a diagonal row of Crafting Stations that is 7 units long.",
            "Can you find a diagonal line of BlockType.CRAFTING_TABLE with a size of 7?",
            "Confirm if a diagonal line made up of the BlockType.CRAFTING_TABLE and with a length of 7 units can be found.",
            "Check for the existence of a Crafting Desk diagonal series with seven items in its sequence."
        ],
        "arguments": create_target_state(BlockType.CRAFTING_TABLE, 7, is_diagonal=True),
        "str_check_lambda": "str_check_lambda: is_line_formed(gd, ix)"
    }
}

