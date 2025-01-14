from craftext.checkers_jax.conditional import conditional_placing
from craftext.scenarios.constants import InventoryItems, BlockType, MediumInventoryItems


one = {
    'simple_conditional_placing_001': {
        'instruction': "Place a furnace on the ground after collecting 3 pieces of wood",
        'instruction_paraphrases': [
            "After gathering three logs, place a kiln on the ground",
            "Collect three wooden blocks and then position a heater on the ground",
            "Once you've collected 3 logs, set up an oven on the terrain",
            "Accumulate three wooden units and put a stove on the ground",
            "Gather three pieces of timber and install a smelter at the designated spot"
        ],
        'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.WOOD, BlockType.FURNACE, 1, 1)
    },
    'simple_conditional_placing_002': {
        'instruction': "Place a furnace on the ground after collecting 3 pieces of wood",
        'instruction_paraphrases': [
            "After gathering three logs, place a kiln on the ground",
            "Collect three wooden blocks and then position a heater on the ground",
            "Once you've collected 3 logs, set up an oven on the terrain",
            "Accumulate three wooden units and put a stove on the ground",
            "Gather three pieces of timber and install a smelter at the designated spot"
        ],
        'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.WOOD, BlockType.FURNACE, 1, 1)
    },
}
    
easy = {}

medium = {}

from craftext.scenarios.parce_dataset import update_previous_dict
easy = update_previous_dict(easy, "../craftext/scenarios/jax_conditional_placing/instructions/train/easy", "conditional_placing")
medium = update_previous_dict(medium, "../craftext/scenarios/jax_conditional_placing/instructions/train/medium", "conditional_placing")
