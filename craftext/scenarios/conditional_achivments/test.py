

from craftext.checkers_jax.achivments import conditional_achivments
import jax
import jax.numpy as jnp
from jax import lax
from enum import Enum


class Achievement(Enum):
    COLLECT_WOOD = 0
    PLACE_TABLE = 1
    EAT_COW = 2
    COLLECT_SAPLING = 3
    COLLECT_DRINK = 4
    MAKE_WOOD_PICKAXE = 5
    MAKE_WOOD_SWORD = 6
    PLACE_PLANT = 7
    DEFEAT_ZOMBIE = 8
    COLLECT_STONE = 9
    PLACE_STONE = 10
    EAT_PLANT = 11
    DEFEAT_SKELETON = 12
    MAKE_STONE_PICKAXE = 13
    MAKE_STONE_SWORD = 14
    WAKE_UP = 15
    PLACE_FURNACE = 16
    COLLECT_COAL = 17
    COLLECT_IRON = 18
    COLLECT_DIAMOND = 19
    MAKE_IRON_PICKAXE = 20
    MAKE_IRON_SWORD = 21

easy_test_parafrased = {
    
    "6. eat_cow_make_stone_pickaxe": {
        "instruction": "Eat a cow and craft a stone pickaxe.",
        "instruction_paraphrases": [
            "Consume beef and craft a stone pickaxe.",
            "Eat steak and forge a stone pickaxe.",
            "Devour cow meat and create a stone pickaxe.",
            "Savor a meal of beef and assemble a stone pickaxe.",
            "Enjoy cooked cow meat and build a stone pickaxe."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.EAT_COW.value, Achievement.MAKE_STONE_PICKAXE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])), 
        
    },
    "8. place_plant_defeat_zombie": {
        "instruction": "Place a plant and defeat a zombie.",
        "instruction_paraphrases": [
            "Plant a tree and eliminate a zombie.",
            "Place a shrub and take down a zombie.",
            "Set greenery into the ground and destroy a zombie.",
            "Drop a sapling and vanquish a zombie.",
            "Position a flower and defeat an undead creature."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.PLACE_PLANT.value, Achievement.DEFEAT_ZOMBIE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "10. place_furnace_make_iron_sword_no_pickaxe": {
        "instruction": "Place a furnace and craft an iron sword but do not make an iron pickaxe.",
        "instruction_paraphrases": [
            "Set up a smelter and forge an iron sword, avoiding a pickaxe.",
            "Place a heater and craft an iron blade without making a pickaxe.",
            "Install a forge and create a sword from iron, skipping pickaxe crafting.",
            "Drop a furnace and construct an iron weapon, avoiding a pickaxe.",
            "Put down a stove and make an iron sword, steering clear of pickaxe production."
        ],
         "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.PLACE_FURNACE.value, Achievement.MAKE_IRON_SWORD.value] else
            -1 if a == Achievement.MAKE_IRON_PICKAXE.value else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "12. defeat_zombie_not_defeat_skeleton": {
        "instruction": "Defeat a zombie but do not kill a skeleton.",
        "instruction_paraphrases": [
            "Eliminate a walker and avoid harming skeletons.",
            "Take down an undead and spare the bone warriors.",
            "Destroy a zombie while leaving skeletons intact.",
            "Vanquish a corpse walker and refrain from attacking skeletons.",
            "Fight and defeat a ghoul, avoiding skeletons."
        ],
         "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.DEFEAT_ZOMBIE.value else
            -1 if a == Achievement.DEFEAT_SKELETON.value else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "14. make_stone_sword_defeat_skeleton": {
        "instruction": "Craft a stone sword and defeat a skeleton.",
        "instruction_paraphrases": [
            "Create a blade from cobblestone and slay a skeleton.",
            "Forge a stone weapon and eliminate a skeletal foe.",
            "Build a sword from rock and destroy a skeleton.",
            "Make a stone sabre and take down a skeleton.",
            "Construct a stone blade and vanquish a bony warrior."
        ],
         "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.MAKE_STONE_SWORD.value, Achievement.DEFEAT_SKELETON.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "16. make_stone_pickaxe_collect_coal": {
        "instruction": "Craft a stone pickaxe and collect coal.",
        "instruction_paraphrases": [
            "Forge a mining tool from stone and mine charcoal.",
            "Build a cobblestone pickaxe and extract black ore.",
            "Create a stone tool and gather coal resources.",
            "Craft a rocky pickaxe and collect fuel ore.",
            "Make a durable pickaxe from stone and harvest coal."
        ],
         "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.MAKE_STONE_PICKAXE.value, Achievement.COLLECT_COAL.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    }
}


easy_test_other_paramets = {
        "7. eat_cow_make_stone_sword": {
        "instruction": "Eat a cow and craft a stone sword.",
        "instruction_paraphrases": [
            "Consume beef and forge a stone blade.",
            "Eat cow meat and create a sword from stone.",
            "Savor beef and construct a stone sword.",
            "Devour a meal from a cow and make a stone sword.",
            "Enjoy beef and build a sword out of stone."
        ],
         "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.EAT_COW.value, Achievement.MAKE_STONE_SWORD.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    }, 
    "9. place_plant_defeat_skeleton": {
        "instruction": "Place a plant and defeat a skeleton.",
        "instruction_paraphrases": [
            "Set a plant into the soil and destroy a skeleton.",
            "Position a plant and eliminate a skeletal enemy.",
            "Plant a green sprout and defeat a skeleton.",
            "Drop a plant and vanquish a skeleton warrior.",
            "Install a plant and take down a bone-clad foe."
        ],
         "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.PLACE_PLANT.value, Achievement.DEFEAT_SKELETON.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
        "11. defeat_skeleton_not_defeat_zombie": {
        "instruction": "Defeat a skeleton but do not kill a zombie.",
        "instruction_paraphrases": [
            "Take down a skeleton and leave zombies alone.",
            "Eliminate a skeleton while avoiding zombies.",
            "Destroy a skeletal enemy but do not engage with zombies.",
            "Fight a skeleton and spare the undead creatures.",
            "Vanquish a skeleton but refrain from fighting zombies."
        ],
         "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.DEFEAT_SKELETON.value else
            -1 if a == Achievement.DEFEAT_ZOMBIE.value else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    }, 
      "13. make_wood_sword_defeat_skeleton": {
        "instruction": "Craft a wooden sword and defeat a skeleton.",
        "instruction_paraphrases": [
            "Make a sword out of wood and destroy a skeleton.",
            "Craft a wooden blade and take down a skeletal enemy.",
            "Forge a wooden weapon and defeat a skeleton warrior.",
            "Build a sword from wood and vanquish a skeleton.",
            "Create a wooden sword and eliminate a skeletal foe."
        ],
         "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.MAKE_WOOD_SWORD.value, Achievement.DEFEAT_SKELETON.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
     "15. make_stone_pickaxe_collect_iron": {
        "instruction": "Craft a stone pickaxe and collect iron.",
        "instruction_paraphrases": [
            "Forge a stone pickaxe and mine some iron.",
            "Create a mining tool from stone and gather iron ore.",
            "Build a durable pickaxe and extract iron from the ground.",
            "Make a stone pickaxe and collect metallic resources.",
            "Craft a stone pickaxe and retrieve iron from a vein."
        ],
         "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.MAKE_STONE_PICKAXE.value, Achievement.COLLECT_IRON.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    }
}