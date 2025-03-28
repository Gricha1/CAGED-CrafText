from craftext.checkers_jax.building import is_square_formed
from craftext.scenarios.constants import BlockType
from craftext.scenarios.parce_dataset import update_previous_dict

easy = {
    "INSTRUCTION_TORCH_5": {
        "instruction": "Construct a direct cross made from torches, each of its sides should consist of 5 blocks.",
        "scenario_checker": 3,
        "instruction_paraphrases": [
            "Build a cross using torches, each arm should contain 5 blocks laid out in a straight line.",
            "Erect a cross with the material of torch, ensuring each limb is direct and made of five blocks.",
            "Set up a cross of torch and make sure every arm is direct and precisely five blocks in length.",
            "Create a directly shaped cross using torches, each side of which should be made up of five blocks.",
            "Assemble a five-block long, direct shaped cross using torch materials."
        ],
        "arguments": create_target_state(BlockType.TORCH, size=5, radius=5, crossType.STRAIGHT),
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_PLANT_5": {
        "instruction": "Form a cross with Plant elements, each side should be 5 units long and in diagonal shape.",
        "scenario_checker": 3,
        "instruction_paraphrases": [
            "Use Plant elements to construct a diagonal cross with each side consisting of 5 units.",
            "Make a cross that is diagonally oriented using Plant items, ensuring each arm of the cross is 5 units in length.",
            "Create a diagonal cross with 5 units on each side, using various forms of Plant elements for construction.",
            "With a Plant item, form a cross that has a side length of 5 units and is oriented diagonally.",
            "Assemble a 5-unit each side diagonal cross using various Plant components."
        ],
        "arguments": create_target_state(BlockType.PLANT, size=5, radius=5, crossType.STRAIGHT),
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_STONE_5": {
        "instruction": "Create a cross of stones with arms of length 5 in any direction.",
        "scenario_checker": 3,
        "instruction_paraphrases": [
            "Form a cross using rocks, ensuring each arm extends 5 units long, and you can orient it in any way.",
            "Configure a collection of pebbles into a cruciform shape, with the length of each limb equally measuring 5 units; the alignment of the cross is up to you.",
            "Using any form of alignment, assemble a cross, consisting of boulders, where each arm stretches out to 5 spaces.",
            "In any given direction, erect an equilateral crucifix shaped configuration from chunks of rock, each arm of which should span 5 units.",
            "Fashion a cross of stones of any orientation where each arm's length equals 5 units."
        ],
        "arguments": create_target_state(BlockType.STONE, size=5, radius=5, crossType.STRAIGHT),
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_PLANT_3": {
        "instruction": "Make a cross shape using plants with a side length of 3 units.",
        "scenario_checker": 3,
        "instruction_paraphrases": [
            "Create a plant-based cross with each side being 3 units long.",
            "Construct a cross figure using flora with each arm spanning 3 units",
            "Use vegetation to form a cross, each of which measures 3 units.",
            "Design a cruciform structure using plants, with each side having a length of 3 units.",
            "Fabricate a cross-shaped figure with greenery having each side of 3 units."
        ],
        "arguments": create_target_state(BlockType.PLANT, size=3, radius=5, crossType.STRAIGHT),
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_STONE_5_DIAGONAL": {
        "instruction": "Create a cross of stones with side size equal to 5 arranged in a diagonal pattern.",
        "scenario_checker": 3,
        "instruction_paraphrases": [
            "Construct a stone cross where each side length is 5 and in diagonal formation.",
            "I would like you to form a cross using rocks with five units on each side in a diagonal orientation.",
            "Build a diagonal cross with pebbles, make sure each side is made up of a total of five units.",
            "Design a boulder-formed cross utilizing a diagonal layout with each arm having a length of 5.",
            "Using rubble, sketch out a cross in a slanted arrangement where each line is composed of five units."
        ],
        "arguments": create_target_state(BlockType.STONE, size=5, radius=7, crossType.STRAIGHT),
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_STONE_7": {
        "instruction": "Construct a cross of stone of side size 7 and with a diagonal shape.",
        "scenario_checker": 3,
        "instruction_paraphrases": [
            "Create a diagonal cross using rock, with each arm being seven units long.",
            "Use pebbles to form a cross diagonally, ensuring each side measures seven units.",
            "Make a diagonal cross by using cobblestones, each arm should be seven units long.",
            "Shape a rock into a cross with seven units on each side, ensuring it's laid out diagonally.",
            "Craft a cross-shaped design using small stones diagonally, and it should span seven units on each side."
        ],
        "arguments": create_target_state(BlockType.STONE, size=7, radius=7, crossType.STRAIGHT),
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_STONE_3": {
        "instruction": "Create a direct cross made of stone with each side measuring 3 blocks.",
        "scenario_checker": 3,
        "instruction_paraphrases": [
            "Craft a straightforward cross shape out of rock with each arm being 3 blocks long.",
            "Please form a rock cross with each of its sides being three units long in a direct shape.",
            "Would you mind making a cross from stone blocks each side of which measures up to three units, and make sure it's not diagonal or combined?",
            "I need you to construct a direct cross using stone material, make sure that the length of each of it sides is exactly three blocks.",
            "Make a formation in a direct cross pattern using stone blocks where each branch of the cross is three units long."
        ],
        "arguments": create_target_state(BlockType.STONE, size=3, radius=5, crossType.STRAIGHT),
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_FIRE_TREE_5": {
        "instruction": "Form a cross of torchlights each side having 5 units and in combined form.",
        "scenario_checker": 3,
        "instruction_paraphrases": [
            "Build a blended cross using torches that is five units long on each side.",
            "Use your luminary torches to construct a cross where each side measures five units and the shape is amalgamated.",
            "Create a fusion design of a cross using illumination torches with each arm spanning five units long.",
            "Five unit long on each side, design a merged cross symbol, using your light torch.",
            "With a combination approach, construct a five-unit long cross structure using fire sticks."
        ],
        "arguments": create_target_state(BlockType.FIRE_TREE, size=5, radius=5, crossType.STRAIGHT),
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_ENCHANTMENT_TABLE_FIRE_3": {
        "instruction": "Make sure you have an enchantment table fire arranged in the form of a cross with a size of 3.",
        "scenario_checker": 3,
        "instruction_paraphrases": [
            "Construct a cross using the fire enchantment table, it should have a length of three.",
            "Place the enchantment table fire units in a direct-shape cross configuration with a size of three.",
            "Can you arrange three of our enchantment table fire units in a cross format?",
            "I want you to form a cross with a side length of three using the fire enchantment table.",
            "With a size of three, form a cross from the units of enchantment table fire."
        ],
        "arguments": create_target_state(BlockType.ENCHANTMENT_TABLE_FIRE, size=3, radius=5, crossType.STRAIGHT),
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_PLANT_7_COMBINED": {
        "instruction": "Make a cross of plant blocks that is 7 blocks on each side and combined shape around you.",
        "scenario_checker": 3,
        "instruction_paraphrases": [
            "Create a cross shape using the plant blocks with each arm having 7 blocks in a combined formation.",
            "Use plant blocks to form a combined cross shape, with each side made up of 7 blocks.",
            "Shape a 7 block per side cross using plant blocks and make sure the shape is combined.",
            "I want you to form a combined cross structure using plant blocks, each side should consists of 7 blocks.",
            "Fabricate a combined cross structure using seven blocks of plant on each side."
        ],
        "arguments": create_target_state(BlockType.PLANT, size=7, radius=7, crossType.STRAIGHT),
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_TORCH_3": {
        "instruction": "Create a cross of torch with size of 3 in a diagonal form",
        "scenario_checker": 3,
        "instruction_paraphrases": [
            "In a diagonal shape, craft a torch cross that measures three units across",
            "Erect a torch cross measuring three units diagonally",
            "Your task is to form a cross with a torch that has a three unit span in a diagonal pattern",
            "Create a diagonal cross with a torch, ensuring that it has a span of three units",
            "In a diagonal fashion, construct a cross out of a torch with a total length of three units"
        ],
        "arguments": create_target_state(BlockType.TORCH, size=3, radius=5, crossType.STRAIGHT),
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_STONE_3_DIAGONAL": {
        "instruction": "Form a cross made of stone with a side size of 3 in a diagonal shape.",
        "scenario_checker": 3,
        "instruction_paraphrases": [
            "Build a stone cross that has 3 units on each side and is arranged diagonally.",
            "Create a cross using stones, ensure it has 3 blocks on each side and is designed diagonally.",
            "I'd like you to construct a cross, diagonally. The material should be stone and the side size should measure three.",
            "Construct a cross with a side length of three using stones and align it in a diagonal direction.",
            "Arrange a cross made from rocks with each side measuring three units, make sure it's set diagonally."
        ],
        "arguments": create_target_state(BlockType.STONE, size=3, radius=5, crossType.STRAIGHT),
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_FURNACE_5": {
        "instruction": "Arrange the blocks in a cross form of furnaces with a size of 5 blocks in diagonal way in the game",
        "scenario_checker": 3,
        "instruction_paraphrases": [
            "I need you to form a cross using furnace blocks, the cross should be diagonal and be made of 5 blocks.",
            "Shape a diagonal cross using 5 blocks, and all blocks should be furnace blocks.",
            "Make a cross within the game using furnaces, ensuring its a diagonal shape and include a total of five units",
            "In the game, create a diagonal cross composed entirely of furnaces, it should have five blocks",
            "Can you arrange furnace blocks in the form of a cross of 5 units in a diagonal form in the game?"
        ],
        "arguments": create_target_state(BlockType.FURNACE, size=5, radius=5, crossType.STRAIGHT),
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_FURNACE_3": {
        "instruction": "Form a cross with a furnace having a side size of three and in a direct shape.",
        "scenario_checker": 3,
        "instruction_paraphrases": [
            "Create a three-sided direct shape cross using the furnace.",
            "With the help of a furnace, could you make a direct cross that has three units on each side?",
            "Construct a cross that is directly shaped and has a side length of three units, utilizing the kiln.",
            "A direct cross of a three on each side needs to be formed, can you create this using the oven?",
            "Centrally arrange the smelter to create a three-unit direct shaped cross."
        ],
        "arguments": create_target_state(BlockType.FURNACE, size=3, radius=5, crossType.STRAIGHT),
        "str_check_lambda": "is_cross_formed(gd, ix)"
    },
    "INSTRUCTION_PLANT_7_DIRECT": {
        "instruction": "Make a cross out of plants with a side length of 7",
        "scenario_checker": 3,
        "instruction_paraphrases": [
            "Form a plant cross of seven units on every side.",
            "Construct a cross using plants with each arm measuring seven units long.",
            "Could you create a flora cross with a total of seven units for every side?",
            "Please generate a vegetation cross with each branch having seven units in length.",
            "I would like to see you create a diagonal cross from greenery with the length of each segment equal to seven units."
        ],
        "arguments": create_target_state(BlockType.PLANT, size=7, radius=5, crossType.STRAIGHT),
        "str_check_lambda": "is_cross_formed(gd, ix)"
    }
}


medium = {}