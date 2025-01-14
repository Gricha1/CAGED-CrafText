
from enum import Enum
from craftext.checkers_jax.achivments import conditional_achivments
from craftext.scenarios.constants import Achievement
import jax
import jax.numpy as jnp
from jax import lax
from enum import Enum

one = {
   "1. collect_wood_place_table": {
        "instruction": "Chop some wood and install a workbench.",
        "instruction_paraphrases": [
             "Gather wood and set down a crafting table.",
             "Harvest timber and position a crafting bench nearby.",
             "Chop some wood and install a workbench.",
             "Retrieve logs and put a crafting table on the ground.",
            "Procure wood and place a crafting table in the area."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_WOOD.value, Achievement.PLACE_TABLE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
        
     "2. collect_wood_place_table": {
        "instruction": "Harvest timber and position a crafting bench nearby.",
        "instruction_paraphrases": [
             "Gather wood and set down a crafting table.",
             "Harvest timber and position a crafting bench nearby.",
             "Chop some wood and install a workbench.",
             "Retrieve logs and put a crafting table on the ground.",
            "Procure wood and place a crafting table in the area."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_WOOD.value, Achievement.PLACE_TABLE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
}
easy = {
    "collect_wood": {
        "instruction": "Collect wood.",
        "instruction_paraphrases": [
            "Harvest logs from nearby trees.",
            "Chop down some timber to gather wood.",
            "Cut a tree to obtain wooden resources.",
            "Retrieve lumber from a fallen tree.",
            "Procure wood by felling trees in the area."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.COLLECT_WOOD.value else 0 for a in  range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "place_table": {
        "instruction": "Place a crafting table.",
        "instruction_paraphrases": [
            "Set down a crafting bench in the area.",
            "Install a crafting workstation at your location.",
            "Position a workbench for crafting nearby.",
            "Drop a table designed for crafting tasks.",
            "Arrange a crafting station in a suitable spot."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.PLACE_TABLE.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "eat_cow": {
        "instruction": "Eat a cow.",
        "instruction_paraphrases": [
            "Consume beef from a butchered cow.",
            "Devour meat obtained from a bovine animal.",
            "Savor a meal made from cow flesh.",
            "Enjoy a dish prepared with beef.",
            "Ingest cow meat for nourishment."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.EAT_COW.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "collect_sapling": {
        "instruction": "Gather a sapling.",
        "instruction_paraphrases": [
            "Pick up a small tree shoot from the ground.",
            "Retrieve a sapling to plant elsewhere.",
            "Harvest a sprouting tree seedling.",
            "Find and collect a young tree sprout.",
            "Gather a tree offspring ready for planting."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.COLLECT_SAPLING.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "collect_drink": {
        "instruction": "Collect a drink.",
        "instruction_paraphrases": [
            "Retrieve a liquid for hydration.",
            "Acquire a beverage to quench thirst.",
            "Find and collect a drinkable resource.",
            "Gather a liquid item suitable for drinking.",
            "Procure a refreshing drink from nearby."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.COLLECT_DRINK.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "make_wood_pickaxe": {
        "instruction": "Craft a wooden pickaxe.",
        "instruction_paraphrases": [
            "Assemble a mining tool made of wood.",
            "Construct a wooden pickaxe for digging.",
            "Fashion a pickaxe out of wooden parts.",
            "Carve and build a wooden mining tool.",
            "Forge a lightweight pickaxe from wood."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.MAKE_WOOD_PICKAXE.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "make_wood_sword": {
        "instruction": "Craft a wooden sword.",
        "instruction_paraphrases": [
            "Forge a blade made from wooden materials.",
            "Carve a wooden sword for protection.",
            "Construct a simple sword using wood.",
            "Create a weapon crafted from timber.",
            "Build a wooden blade for self-defense."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.MAKE_WOOD_SWORD.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "place_plant": {
        "instruction": "Place a plant.",
        "instruction_paraphrases": [
            "Set a plant into the soil.",
            "Position a green sprout in the ground.",
            "Plant a botanical seedling in the area.",
            "Install a plant in a sunny location.",
            "Place a flower or shrub in a chosen spot."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.PLACE_PLANT.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "defeat_zombie": {
        "instruction": "Defeat a zombie.",
        "instruction_paraphrases": [
            "Eliminate an undead creature lurking nearby.",
            "Vanquish a wandering zombie in combat.",
            "Destroy a rotting foe roaming the area.",
            "Take down a zombie using any weapon.",
            "Overcome a night-stalking undead being."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.DEFEAT_ZOMBIE.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "collect_stone": {
        "instruction": "Collect stone.",
        "instruction_paraphrases": [
            "Mine stone blocks from a rocky surface.",
            "Break apart rocks to gather stone.",
            "Harvest stone materials from the ground.",
            "Retrieve stone fragments from nearby boulders.",
            "Extract useful stone for crafting purposes."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.COLLECT_STONE.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "place_stone": {
        "instruction": "Place a stone block.",
        "instruction_paraphrases": [
            "Set a block of stone in its place.",
            "Position a solid stone block on the ground.",
            "Install a stone cube where needed.",
            "Arrange a block of stone in the area.",
            "Place a stone slab in the desired spot."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.PLACE_STONE.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "eat_plant": {
        "instruction": "Eat a plant.",
        "instruction_paraphrases": [
            "Consume a green plant for sustenance.",
            "Nibble on vegetation to regain strength.",
            "Eat a leaf-based food source.",
            "Chew on a herbaceous snack for energy.",
            "Devour a plant to satisfy your hunger."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.EAT_PLANT.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "defeat_skeleton": {
        "instruction": "Defeat a skeleton.",
        "instruction_paraphrases": [
            "Destroy a bony adversary roaming nearby.",
            "Vanquish a skeletal warrior in combat.",
            "Eliminate a skeleton using your weapon.",
            "Overpower a bone-clad enemy in battle.",
            "Take down a skeletal creature in the area."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.DEFEAT_SKELETON.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "make_stone_pickaxe": {
        "instruction": "Craft a stone pickaxe.",
        "instruction_paraphrases": [
            "Forge a sturdy pickaxe from stone.",
            "Construct a durable mining tool using rocks.",
            "Create a pickaxe built from stone materials.",
            "Carve a reliable pickaxe from stone.",
            "Assemble a heavy-duty stone pickaxe."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.MAKE_STONE_PICKAXE.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
     "wake_up": {
        "instruction": "Wake up.",
        "instruction_paraphrases": [
            "Rise from your sleep.",
            "Get out of bed and start your day.",
            "Awaken and prepare for action.",
            "Stand up after resting.",
            "Shake off your slumber and wake up."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.WAKE_UP.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "place_furnace": {
        "instruction": "Place a furnace.",
        "instruction_paraphrases": [
            "Set up a furnace for smelting.",
            "Install a furnace at your location.",
            "Position a smelter in the area.",
            "Drop a furnace for crafting needs.",
            "Put a furnace on the ground to use."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.PLACE_FURNACE.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "collect_coal": {
        "instruction": "Collect coal.",
        "instruction_paraphrases": [
            "Mine coal from the ground.",
            "Harvest coal for smelting purposes.",
            "Retrieve black ore to use as fuel.",
            "Extract coal from a nearby deposit.",
            "Gather some coal for crafting."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.COLLECT_COAL.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "collect_iron": {
        "instruction": "Collect iron.",
        "instruction_paraphrases": [
            "Mine iron ore from the earth.",
            "Gather raw iron for forging.",
            "Extract iron from a nearby vein.",
            "Retrieve iron ore for crafting tools.",
            "Harvest metallic resources for use."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.COLLECT_IRON.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "collect_diamond": {
        "instruction": "Collect a diamond.",
        "instruction_paraphrases": [
            "Mine a shiny diamond from the ground.",
            "Retrieve a precious gem from a deposit.",
            "Harvest a sparkling jewel for crafting.",
            "Find and collect a valuable diamond.",
            "Extract a rare diamond from a hidden vein."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.COLLECT_DIAMOND.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "make_iron_pickaxe": {
        "instruction": "Craft an iron pickaxe.",
        "instruction_paraphrases": [
            "Forge a durable pickaxe using iron.",
            "Construct a mining tool from iron ingots.",
            "Create a pickaxe made of iron.",
            "Assemble an iron pickaxe for digging.",
            "Build a reliable pickaxe forged from iron."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.MAKE_IRON_PICKAXE.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "make_iron_sword": {
        "instruction": "Craft an iron sword.",
        "instruction_paraphrases": [
            "Forge a blade out of iron.",
            "Create a sword using iron ingots.",
            "Build a weapon made of iron.",
            "Construct a sharp sword forged from iron.",
            "Fashion an iron sword for combat."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.MAKE_IRON_SWORD.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    }
}


from craftext.scenarios.parce_dataset import update_previous_dict
easy = update_previous_dict(easy, "../craftext/scenarios/jax_conditional_achivments/instructions/train/easy", "achivments")
    

if __name__ == "__main__":
    import json
    instructions = []
    for key in easy.keys():
        instructions.append(easy[key]['instruction'])
        instructions += easy[key]['instruction_paraphrases']
    with open("instructions_achivments_easy_train.json", "w", encoding="utf-8") as json_file:
        json.dump(instructions, json_file, ensure_ascii=False, indent=4)