
from enum import Enum
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

    # MAKE_ARROW = 22
    # MAKE_TORCH = 23
    # PLACE_TORCH = 24

    # COLLECT_SAPPHIRE = 54
    # COLLECT_RUBY = 59
    # MAKE_DIAMOND_PICKAXE = 60
    # MAKE_DIAMOND_SWORD = 25
    # MAKE_IRON_ARMOUR = 26
    # MAKE_DIAMOND_ARMOUR = 27

    # ENTER_GNOMISH_MINES = 28
    # ENTER_DUNGEON = 29
    # ENTER_SEWERS = 30
    # ENTER_VAULT = 31
    # ENTER_TROLL_MINES = 32
    # ENTER_FIRE_REALM = 33
    # ENTER_ICE_REALM = 34
    # ENTER_GRAVEYARD = 35

    # DEFEAT_GNOME_WARRIOR = 36
    # DEFEAT_GNOME_ARCHER = 37
    # DEFEAT_ORC_SOLIDER = 38
    # DEFEAT_ORC_MAGE = 39
    # DEFEAT_LIZARD = 40
    # DEFEAT_KOBOLD = 41
    # DEFEAT_KNIGHT = 65
    # DEFEAT_ARCHER = 66
    # DEFEAT_TROLL = 42
    # DEFEAT_DEEP_THING = 43
    # DEFEAT_PIGMAN = 44
    # DEFEAT_FIRE_ELEMENTAL = 45
    # DEFEAT_FROST_TROLL = 46
    # DEFEAT_ICE_ELEMENTAL = 47
    # DAMAGE_NECROMANCER = 48
    # DEFEAT_NECROMANCER = 49

    # EAT_BAT = 50
    # EAT_SNAIL = 51

    # FIND_BOW = 52
    # FIRE_BOW = 53

    # LEARN_FIREBALL = 55
    # CAST_FIREBALL = 56
    # LEARN_ICEBALL = 57
    # CAST_ICEBALL = 58

    # OPEN_CHEST = 61
    # DRINK_POTION = 62
    # ENCHANT_SWORD = 63
    # ENCHANT_ARMOUR = 64

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
    },
    "1. collect_wood_place_table": {
        "instruction": "Collect wood and place a crafting table.",
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
    "2. make_wood_pickaxe_collect_stone": {
        "instruction": "Craft a wooden pickaxe and collect stone.",
        "instruction_paraphrases": [
            "Create a wooden pickaxe and mine stone.",
            "Assemble a wood pickaxe and gather stone.",
            "Craft a mining tool from wood and extract stone.",
            "Build a wooden pickaxe and harvest stone blocks.",
            "Forge a wooden pickaxe and collect rocky materials."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.MAKE_WOOD_PICKAXE.value, Achievement.COLLECT_STONE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "3. collect_drink_eat_cow": {
        "instruction": "Collect a drink and eat a cow.",
        "instruction_paraphrases": [
            "Retrieve a beverage and consume beef.",
            "Find a drink and eat cow meat.",
            "Collect a liquid and savor cow flesh.",
            "Gather a drinkable item and devour beef.",
            "Acquire a drink and enjoy a meal from a cow."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_DRINK.value, Achievement.EAT_COW.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "4. place_stone_wake_up": {
        "instruction": "Place a stone block and wake up.",
        "instruction_paraphrases": [
            "Set down a stone block and rise from sleep.",
            "Position a stone and awaken.",
            "Place a block of stone and stand up after resting.",
            "Install a stone block and shake off slumber.",
            "Drop a stone block and prepare for the day."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.PLACE_STONE.value, Achievement.WAKE_UP.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "5. place_furnace_place_stone": {
        "instruction": "Place a furnace and a stone block.",
        "instruction_paraphrases": [
            "Set up a furnace and position a stone block.",
            "Install a smelter and place down a stone.",
            "Drop a furnace and a block of stone.",
            "Put a furnace and a stone in their spots.",
            "Arrange a furnace and a stone block nearby."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.PLACE_FURNACE.value, Achievement.PLACE_STONE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    
    "6. eat_cow_make_stone_pickaxe": {
        "instruction": "Eat a cow and craft a stone pickaxe.",
        "instruction_paraphrases": [
            "Consume beef and create a stone pickaxe.",
            "Eat cow meat and forge a stone mining tool.",
            "Savor a meal from a cow and craft a pickaxe from stone.",
            "Devour beef and assemble a stone pickaxe.",
            "Have a cow-based dish and make a stone pickaxe."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.EAT_COW.value, Achievement.MAKE_STONE_PICKAXE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    "8. place_plant_defeat_zombie": {
        "instruction": "Place a plant and defeat a zombie.",
        "instruction_paraphrases": [
            "Set a plant into the ground and eliminate a zombie.",
            "Position a plant and destroy an undead creature.",
            "Plant a shrub and take down a zombie.",
            "Drop a plant and vanquish a wandering zombie.",
            "Install a plant and defeat a night-stalking zombie."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.PLACE_PLANT.value, Achievement.DEFEAT_ZOMBIE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    },
    
    "10. place_furnace_make_iron_sword_no_pickaxe": {
        "instruction": "Place a furnace and craft an iron sword but do not make an iron pickaxe.",
        "instruction_paraphrases": [
            "Set up a furnace and forge an iron blade, avoiding pickaxe crafting.",
            "Place a furnace and create an iron sword, refraining from making a pickaxe.",
            "Install a furnace and craft a sword from iron, ensuring no pickaxe is made.",
            "Drop a furnace and construct an iron weapon, skipping pickaxe creation.",
            "Put down a furnace and make an iron sword, avoiding the pickaxe blueprint."
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
            "Take down a zombie and avoid fighting skeletons.",
            "Eliminate a zombie while sparing the skeletons.",
            "Destroy a zombie but refrain from attacking skeletons.",
            "Fight and defeat a zombie, leaving skeletons untouched.",
            "Vanquish a zombie while avoiding any conflict with skeletons."
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
            "Forge a stone blade and eliminate a skeletal enemy.",
            "Create a weapon from stone and take down a skeleton.",
            "Build a stone sword and vanquish a bone-clad foe.",
            "Make a sword out of stone and defeat a skeleton warrior.",
            "Construct a stone sword and destroy a skeleton."
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
            "Forge a stone pickaxe and mine coal.",
            "Build a pickaxe from stone and harvest coal.",
            "Create a durable mining tool and extract coal deposits.",
            "Craft a stone pickaxe and gather fuel for smelting.",
            "Make a stone pickaxe and collect black ore."
        ],
        "str_check_lambda":"",
        "check_lambda": lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.MAKE_STONE_PICKAXE.value, Achievement.COLLECT_COAL.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))
    }
}
    
