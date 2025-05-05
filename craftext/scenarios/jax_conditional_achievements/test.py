

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
    },

'INSTRUCTION_2': \
        {
            'instruction': "Collect a sapling and some iron in the game.",
            'instruction_paraphrases': [
                "Grab a sapling and acquire some iron while you're playing.",
                "When you are in-game, be sure to gather a sapling and some iron.",
                "Ensure you have obtained a sapling and iron in your possession during the gameplay.",
                "During your course of play, it's crucial to pick up a sapling, and don't forget to collect some iron as well.",
                "While engaging in the game, it's of utmost importance to procure a sapling and subsequently amass some iron."
            ],
            'check_lambda': lambda gd, ix: conditional_achivments(gd, \
            jnp.array([1 if a in [Achievement.COLLECT_SAPLING.value, Achievement.COLLECT_IRON.value] else 0 \
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)])),
            'str_check_lambda': "conditional_achivments(gd, \
            jnp.array([1 if a in [Achievement.COLLECT_SAPLING.value, Achievement.COLLECT_IRON.value] else 0 \
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
        }
   , 

    'EAT_PLANT': \
    {
        'instruction': "Please consume a plant, while making sure that you have not ingested a cow.",
        'instruction_paraphrases': [
            "Could you devour some green vegetation? Remember, you mustn't have eaten any bovine creature.",
            "Absorb nutrients from a herb, on one condition - that you've refrained from partaking of any cow meat.",
            "I'd like you to eat a plant. But never should you have dined on a cow.",
            "Make sure to consume some plant life, given that you have not indulged in beef.",
            "You ought to be consuming vegetation, provided that beef was not included in your diet."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.EAT_PLANT.value else
            -1 if a == Achievement.EAT_COW.value else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': "lambda gd, ix: conditional_achivments(gd, jnp.array([1 if a == Achievement.EAT_PLANT.value else -1 if a == Achievement.EAT_COW.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
    }, 

'INSTRUCTION_41': 
        {'instruction': "Start by waking up, then gather some wood, and finally place a stone.",
        'instruction_paraphrases': [
            "Awake first, collect some timber and lastly place a boulder.",
            "Begin with awakening, proceed with lumber collection, and conclude with rocky placement.",
            "First, rise from sleep, then amass a pile of logs, and in the end place a rock.",
            "Initiate with a wake-up, continue by harvesting wood, wrap up with setting down a piece of stone.",
            "Commence by stirring from slumber, accumulate wood pieces-next, finish off with a rock placement."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.WAKE_UP.value, Achievement.COLLECT_WOOD.value, Achievement.PLACE_STONE.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.WAKE_UP.value, Achievement.COLLECT_WOOD.value, Achievement.PLACE_STONE.value] else 0 for a in range(Achievement.PLACE_STONE.value+1)]))"
        },

        'INSTRUCTION_5': \
        {
            'instruction': "Collect some logs, place a boulder and craft a stone pickaxe",
            'instruction_paraphrases': [
                "Grab some timber, put down a pebble and manufacture a stone axe",
                "Amass some firewood, set a rock and assemble a stone pick",
                "Procure some lumber, position a stone and fabricate a stone cutter",
                "Accumulate wood material, position a cobblestone and construct an axe made of stone",
                "Get together blocks of wood, install a piece of rock and put together a miner's tool made of stone"
            ],
            'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
                1 if a in [Achievement.COLLECT_WOOD.value, Achievement.PLACE_STONE.value, Achievement.MAKE_STONE_PICKAXE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
            ])),
            'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_WOOD.value, Achievement.PLACE_STONE.value, Achievement.MAKE_STONE_PICKAXE.value] else 0 for a in range(Achievement.MAKE_STONE_PICKAXE.value+1)]))"
        }
   , 

        'INSTRUCTION_6': 
        {
            'instruction': "Place a furnace and make an iron sword. Do not make an iron pickaxe.",
            'instruction_paraphrases': [
                "Please set down a stove and forge an iron weapon, but don't create an iron mining tool.",
                "Can you establish a heating device then construct a sword made of iron? But refrain from producing an iron tool for digging.",
                "You are tasked to install a furnace and fabricate a weaponry made of iron, however, manufacturing an iron pick is not advisable.",
                "It is required of you to position a melting device properly, subsequently create a bladed weapon using iron, yet, an absolute prohibition on the creation of an excavating tool made of iron is imposed.",
                "In this quest, you are to effectuate a successful placement of a kiln and produce a defensive instrument composed mainly of iron. However, making a hand-held mining instrument out of iron is strictly prohibited."
            ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
                1 if a in [Achievement.PLACE_FURNACE.value, Achievement.MAKE_IRON_SWORD.value] else
                -1 if a == Achievement.MAKE_IRON_PICKAXE.value else 0
                for a in range(Achievement.MAKE_IRON_SWORD.value+1)
            ])),
        'str_check_lambda':"conditional_achivments(gd, jnp.array([1 if a in [Achievement.PLACE_FURNACE.value, Achievement.MAKE_IRON_SWORD.value] else -1 if a == Achievement.MAKE_IRON_PICKAXE.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
        } 
   , 

    'INSTRUCTION_7': 
    {
        'instruction': "Ensure that you plant a sapling, but don't create a stone sword.",
        'instruction_paraphrases': 
        [
            "I want you to place a seedling but refrain from forging a sword made of stone",
            "Your task is to plant a sapling, while avoiding the creation of a stone-bladed weapon.",
            "Make sure to cultivate a tree sprout but do not manufacture a fighting tool crafted from rock.",
            "Your responsibility is to cultivate a small tree, and ensure you do not fashion a weapon from stone.",
            "It is crucial you plant a diminutive tree and take pains to avoid producing a lethal implement from stone."
        ],
        
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array(
            [1 if a == Achievement.COLLECT_SAPLING.value else
            -1 if a == Achievement.MAKE_STONE_SWORD.value else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
            ]
        )
        ),
        
   'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a == Achievement.COLLECT_SAPLING.value else -1 if a == Achievement.MAKE_STONE_SWORD.value else 0 for a in range(Achievement.MAKE_STONE_SWORD.value+1)]))"
    }, 

    'INSTRUCTION_81': {
    
        'instruction': "Collect a beverage, place some flora, and then defeat a skeleton.",
        
        'instruction_paraphrases': [
        
            "Gather a drink, grow a plant and conquer a skeleton", 
            
            "Procure a liquid refreshment, set down a vegetation and eradicate a skeleton",
            
            "Amass a beverage, position a sprout, and vanquish a bony figure",
            
            "Accumulate a potable, fix a sapling and triumph over a skeletal adversary",
            
            "Acquire an ingestible fluid, establish a shrubbery and outdo a boney antagonist"
        ],
        
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
        
            1 if a in [Achievement.COLLECT_DRINK.value, Achievement.PLACE_PLANT.value, Achievement.DEFEAT_SKELETON.value] else 0
        
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        
        ])),
        
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_DRINK.value, Achievement.PLACE_PLANT.value, Achievement.DEFEAT_SKELETON.value] else 0 for a in range(Achievement.DEFEAT_ZOMBIE.value+1)]))"
    }, 

        'INSTRUCTION_9': \
        {
            'instruction': "Gather some timber and then vanquish a zombie.",
            'instruction_paraphrases': [
                "Collect a bit of wood and afterward eliminate a zombie.",
            
                "Assemble logs then defeat a living dead.",
                
                "Amass some lumber and subsequently slay a ghoul.",
                
                "Accumulate chunks of trees and next crush a monster.",
                "Round up some wooden material and subsequently conquer the undead."
            ],
            'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array(
                [ 1 if a in [Achievement.COLLECT_WOOD.value, Achievement.DEFEAT_ZOMBIE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1) ])),
            'str_check_lambda': "conditional_achivments(gd, jnp.array([ 1 if a in [Achievement.COLLECT_WOOD.value, Achievement.DEFEAT_ZOMBIE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1) ]))"
  
        }
   , 

        'INSTRUCTION_10': 
        {
            'instruction': "You must wake up. Then, you need to defeat a zombie and lastly, fetch a drink for yourself.",
            'instruction_paraphrases': [
                "First, you have to get out of bed. After that, go defeat a zombie. Finally, go acquire a drink.",
                "Your first task is to wake up. Once you have done that, your next job is to destroy a zombie. After that is completed, obtain a beverage for your consumption.",
                "Let's start by getting up from sleep. The next thing on your list is to handle a zombie. Finish up by getting yourself something to drink.",
                "Wake up from your slumber first, then take on a zombie. After you have accomplished those tasks, find a drink for yourself.",
                "Your journey begins when you awaken. After that, go ahead and confront a zombie. End your journey with securing a drink."
            ],
            'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array(
                [
                    1 if a in [Achievement.WAKE_UP.value, Achievement.DEFEAT_ZOMBIE.value, Achievement.COLLECT_DRINK.value] else 0
                    for a in range(Achievement.MAKE_IRON_SWORD.value + 1)
                ]
            )),
            'str_check_lambda': "conditional_achivments(gd, jnp.array(\
                [\
                    1 if a in [Achievement.WAKE_UP.value, Achievement.DEFEAT_ZOMBIE.value, Achievement.COLLECT_DRINK.value] else 0\
                    for a in range(Achievement.MAKE_IRON_SWORD.value + 1)\
                ]\
            ))"
        }
   , 

    'INSTRUCTION_11': {
        'instruction': "Don't create an Iron Sword. You need to, however, craft a Stone Sword and gather a Sapling.",
        'instruction_paraphrases': [
            "Avoid making an Iron Sword, but be sure to forge a Stone Blade and pick up a Sprout.",
            "Without constructing an Iron Smasher, you should still assemble a Pebble Cutter and obtain a Seedling.",
            "Make sure not to form an Iron Saber, yet definitely build a Boulder Slicer and accumulate a Seed.",
            "Steer clear of creating a Sword of Iron, yet it's crucial to manufacture a Rock Slicer and gather a Shoot.",
            "Refrain from producing a Blade of Iron; still, you must create a Sword out of Stone and capture a Plant Bud."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.MAKE_STONE_SWORD.value, Achievement.COLLECT_SAPLING.value] else
            -1 if a == Achievement.MAKE_IRON_SWORD.value else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.MAKE_STONE_SWORD.value, Achievement.COLLECT_SAPLING.value] else-1 if a == Achievement.MAKE_IRON_SWORD.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
    }, 

    'INSTRUCTION_12': {
        'instruction': "Gather some iron, make a wooden pickaxe and defeat a skeleton.",
        'instruction_paraphrases': [
            "Find some iron ore, craft a wooden pickaxe and vanquish a skeleton.",
            "Get hold of some iron, fabricate a pickaxe of wood and conquer a skeleton enemy.",
            "Accumulate some iron, construct a pickaxe made of wood and obliterate a skeleton.",
            "Procure some iron resources, build a wooden tool called pickaxe and eliminate a skeletal adversary.",
            "Requisition some iron mineral, engineer a pickaxe utilizing wood and triumph over a bony antagonist."
        ],
        'check_lambda': 
        lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_IRON.value, Achievement.MAKE_WOOD_PICKAXE.value, Achievement.DEFEAT_SKELETON.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda':"conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_IRON.value, Achievement.MAKE_WOOD_PICKAXE.value, Achievement.DEFEAT_SKELETON.value] else 0 for a in range(Achievement.DEFEAT_SKELETON.value+1)]))"
        }
   , 

    'INSTRUCTION_13 1': {
        'instruction': "Collect both wood and stone in the game.",
        'instruction_paraphrases': [
            "Gather wood and stone resources in your gameplay.",
            "During your game, ensure you have gathered both timber and rocks.",
            "Make sure to collect resources like lumber and boulders while playing game.",
            "In the game, accumulate timberwood items and stone elements.",
            "Carry out an action to collect both woody materials and stoneworks during your game session."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_WOOD.value, Achievement.COLLECT_STONE.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda':"conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_WOOD.value, Achievement.COLLECT_STONE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
    }, 

    'INSTRUCTION_16': {
        'instruction': "Craft a wooden sword and cultivate a plant",
        'instruction_paraphrases': [
            "Make a wooden sword and grow a plant",
            "Fabricate a timber sword and breed a plant",
            "Manufacture an arboreal blade and plant a seedling",
            "Construct a lumber broadsword and develop some vegetation",
            "Create a wooden cutting edge weapon and nurture a botanical specimen"
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([1 if a in [Achievement.MAKE_WOOD_SWORD.value, Achievement.PLACE_PLANT.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.MAKE_WOOD_SWORD.value, Achievement.PLACE_PLANT.value] else 0 for a in range(Achievement.MAKE_WOOD_SWORD.value+1)]))"
    }, 

    'INSTRUCTION_17': \
    {
        'instruction': "Please gather rock and then make an iron sword.",
        'instruction_paraphrases': [
            "Collect some stone first and then forge a sword made of iron.",
            "Could you acquire some pebbles first and afterwards create an iron blade?",
            "Initially amass some rock material and subsequently construct a weapon with iron",
            "I would appreciate if you could gather boulders initially, following by the task of forming a sword using iron",
            "Is it possible for you to initially accumulate some stone and subsequently proceed with the forging of a weapon made from iron?"
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_STONE.value, Achievement.MAKE_IRON_SWORD.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value + 1)
        ])),
        
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_STONE.value, Achievement.MAKE_IRON_SWORD.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value + 1)]))",
    }, 
'INSTRUCTION_18': 
        {'instruction': "It's essential that you collect some diamonds, gather up some iron, and feed yourself with some cow meat.",
        'instruction_paraphrases': [
            "Don't forget to mine some diamonds, gather a few iron ores and eat the meat of a cow.",
            "To survive, make sure you mine diamonds, collect iron ores, and also eat beef.",
            "You need to look out for diamonds to mine, collect some iron and of course, feed on cattle meat.",
            "It's vital that you mine some precious diamonds, search for and gather iron ores around, also remember you'll need to eat beef to nourish yourself.",
            "You'll need to ensure that you heavily mine a lot of diamond ores, engage in serious searches for iron ores and also, partake in feeding on the succulent flesh of cows."
            ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_DIAMOND.value, Achievement.COLLECT_IRON.value, Achievement.EAT_COW.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)])),
        'str_check_lambda':"conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_DIAMOND.value, Achievement.COLLECT_IRON.value, Achievement.EAT_COW.value] else 0 for a in range(Achievement.PLACE_FURNACE.value+1)]))"
    } ,
    'INSTRUCTION_19': \
    {
        'instruction': "Create an axe made of wood and defeat a bony monster.",
        'instruction_paraphrases': [
            "Conquer a skeleton after crafting a timber hatchet.",
            "Vanquish a bone creature post manufacturing a lumber splitting tool.",
            "Overcome a skeletal adversary subsequent to the fabrication of a log splitter.",
            "Triumph over an osteal opponent following the development of a woodcutter's implement.",
            "Subjugate an exoskeletal entity subsequent to the production of a timber chopper."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.MAKE_WOOD_PICKAXE.value, Achievement.DEFEAT_SKELETON.value] else
            0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        
        'str_check_lambda': """conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.MAKE_WOOD_PICKAXE.value, Achievement.DEFEAT_SKELETON.value] else
            0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))"""
    }, 

        'INSTRUCTION_20_ACHIEVEMENT_DEFZOMBIE_EATPLANT': 
        {
            'instruction': "Defeat a zombie and then eat a plant.",
            'instruction_paraphrases': [
                "Take down a undead monster and consume some flora.",
                "Successfully defeat a walking corpse and follow it up by consuming plant-based nourishment.",
                "Battle and vanquish a zombie, afterwards partake in eating a plant.",
                "Combat with a revenant, bring it down and subsequently ingest a botanical item.",
                "Engage in a fight with a living dead creature, outfighting it, following which ingest green fodder."
            ],
            'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
                    1 if a in [Achievement.DEFEAT_ZOMBIE.value, Achievement.EAT_PLANT.value] 
                    else 0 for a in range(Achievement.MAKE_IRON_SWORD.value + 1)
                ])),
            'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.DEFEAT_ZOMBIE.value, Achievement.EAT_PLANT.value] else 0 for a in range(Achievement.EAT_PLANT.value + 1)]))"
        }, 

        'INSTRUCTION_21': \
            {
                'instruction': "You need to collect some logs and mine some iron.",
                'instruction_paraphrases': [
                    "You have to gather a few pieces of lumber and mine a bit of iron.",
                    "Try amassing several chunks of timber along with a few bits of iron ore.",
                    "You're required to accumulate a quantity of wood and obtain a decent amount of iron.",
                    "The task involves getting hold of some wood from trees and unearthing certain amounts of iron deposits.",
                    "Acquire a significant volume of tree-derived materials and excavate considerable traces of the metallic element iron."
                ],
                'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
                    1 if a in [Achievement.COLLECT_WOOD.value, Achievement.COLLECT_IRON.value] else 0
                    for a in range(Achievement.MAKE_IRON_SWORD.value+1)
                ])),
                'str_check_lambda':"conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_WOOD.value, Achievement.COLLECT_IRON.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
            }
   , 

    'INSTRUCTION_22': 
    {
        'instruction': "You need to consume beef from a bovine animal and gather some metal minerals.",
        'instruction_paraphrases': [
            "May you please eat some cow's meat and accumulate some iron ores.",
            "Could you kindly devour cow-flesh and assemble a collection of iron minerals?",
            "You have to feast upon bovine and accrue certain amounts of ferrum.",
            "I require you to ingest some beef and amass a few chunks of iron.",
            "The task involves feeding on flesh of a cow and mining to accumulate iron ores."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.EAT_COW.value, Achievement.COLLECT_IRON.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': "conditional_achivments(...)"
    }, 

        'INSTRUCTION_24_1': {
            'instruction': "Gather coal, iron and eat a plant",
            'instruction_paraphrases': [
                "Collect some coal and iron, and consume a plant",
                "Mine for coal and iron, then eat some vegetables",
                "Procure some coal and iron, following with plant consumption",
                "Accrue deposits of coal and iron, then partake of some vegetation",
                "Assemble an aggregation of coal and iron, before indulging in some herbivorous tendencies"
            ],
            'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
                1 if a in [Achievement.COLLECT_COAL.value, Achievement.COLLECT_IRON.value, Achievement.EAT_PLANT.value] else 0
                for a in range(Achievement.MAKE_IRON_SWORD.value+1)
            ])),
            'str_check_lambda': "'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_COAL.value, Achievement.COLLECT_IRON.value, Achievement.EAT_PLANT.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
        }
   , 

    'INSTRUCTION_25': 
    {
        'instruction': "Create a stone pickaxe and install a furnace for use.",
        'instruction_paraphrases': [
            "Fabricate a pickaxe made of stone and put together a furnace.",
            "Make a stone pickaxe and have a furnace set up.",
            "Concoct a rock pickaxe and get a furnace installed.",
            "Craft a cobblestone pickaxe and establish a kiln.",
            "Construct a stone cutter and erect a furnace."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
                    1 if a in [Achievement.PLACE_FURNACE.value, Achievement.MAKE_STONE_PICKAXE.value] else 0
                    for a in range(Achievement.MAKE_IRON_SWORD.value+1)])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.PLACE_FURNACE.value, Achievement.MAKE_STONE_PICKAXE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
    }, 

        'INSTRUCTION_27_1': {
            'instruction': "Eat a plant, craft a stone pickaxe and then eat another plant",
            'instruction_paraphrases': [
                "Devour a plant, construct a stone pickaxe and munch another plant",
                "Consume some greenery, forge a stone digger, and nibble on another vegetable",
                "Have a plant for a snack, fabricate a stone cutter, and feast on an additional flora",
                "Ingest flora, assemble a rock digger, and partake in extra botany-based nourishment",
                "Swallow greens, build a pebble chopping tool, and chow down another vegetative matter"
            ],
            'check_lambda': lambda gd, ix: conditional_achivments(gd,
                                                             jnp.array([1 if a in [
                                                                 Achievement.EAT_PLANT.value, Achievement.MAKE_STONE_PICKAXE.value, Achievement.EAT_PLANT.value] else 0 for a in
                                                                        range(Achievement.MAKE_IRON_SWORD.value + 1)])),
            'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [ Achievement.EAT_PLANT.value, Achievement.MAKE_STONE_PICKAXE.value, Achievement.EAT_PLANT.value] else 0 for a in range(Achievement.MAKE_STONE_PICKAXE.value + 1)]))"
        }
   , 
        'INSTRUCTION_29':
            {
                'instruction': "Make sure you have placed the furnace and not built an iron pickaxe. You can make the iron sword though.",
                'instruction_paraphrases': [
                    "Ensure the furnace is set but do not construct a pickaxe of iron. Feel free to create an iron sword.",
                    "It's crucial to have established the furnace, but don't bother forming an iron pickaxe. You have permission to synthesize an iron sword.",
                    "Having a furnace in place is necessary and avoid crafting an iron pickaxe, however, the creation of an iron sword is allowed.",
                    "Confirm the furnace installation but refrain from constructing an iron pickaxe. Constructing an iron cutlass is allowable though.",
                    "The furnace should be positioned appropriately, but the fabrication of an iron pickaxe should not be executed. However, forming an iron epee is permissible."
                ],
                'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
                    1 if a in [Achievement.PLACE_FURNACE.value, Achievement.MAKE_IRON_SWORD.value] else
                    -1 if a == Achievement.MAKE_IRON_PICKAXE.value else 0
                    for a in range(Achievement.MAKE_IRON_SWORD.value+1)
                ])),
                'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.PLACE_FURNACE.value, Achievement.MAKE_IRON_SWORD.value] else -1 if a == Achievement.MAKE_IRON_PICKAXE.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
        }
   , 

    'INSTRUCTION_30 1': {
        'instruction': "Get a drink but don't place a workbench.",
        'instruction_paraphrases': [
            "Obtain a beverage, avoid setting up a table.",
            "Procure a liquid refreshment but restrain from furnishing a desk.",
            "Secure a thirst-quencher without positioning a work station.",
            "Acquire an aqueous solution and refrain from locating a worktable.",
            "Apprehend a drink though forbear from establishing a workstation."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_DRINK.value] else
            -1 if a == Achievement.PLACE_TABLE.value else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([ \
            1 if a in [Achievement.COLLECT_DRINK.value] else \
            -1 if a == Achievement.PLACE_TABLE.value else 0 \
            for a in range(Achievement.MAKE_IRON_SWORD.value+1) \
        ]))"
    }, 

        'INSTRUCTION_31_1':
            {
                'instruction': "Collect some coal, place a plant and gather a drink.",
                'instruction_paraphrases': [
                    "Retrieve some coal, position a plant and assemble a drink.",
                    "Garner some bits of coal, establish a plant and fetch a beverage.",
                    "Acquire a number of coals, situate a flora and collect a gulp.",
                    "Obtain pieces of coal, set down a greenery and bring together a liquid nourishment.",
                    "Accumulate deposits of coal, plant and secure a thirst-quenching liquid."
                ],
                'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
                    1 if a in [Achievement.COLLECT_COAL.value, Achievement.PLACE_PLANT.value, Achievement.COLLECT_DRINK.value] else 0
                    for a in range(Achievement.MAKE_IRON_SWORD.value + 1)
                ])),
                'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_COAL.value, Achievement.PLACE_PLANT.value, Achievement.COLLECT_DRINK.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value + 1)]))"
            }
   , 

    'DefeatZombie_PlaceTable': \
    {
        'instruction': "Defeat a zombie and then place a table.",
        'instruction_paraphrases': 
        [
            "Kill a zombie and set up a desk.",
            "Eradicate the zombie and position a workbench.",
            "Eliminate one of the undead and establish a table.",
            "Overcome the walking dead and install a tabletop.",
            "Annihilate a monster from hell and put in place a work surface."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array(
        [
            1 if a in [Achievement.DEFEAT_ZOMBIE.value, Achievement.PLACE_TABLE.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value + 1)
        ])),
        'str_check_lambda':"'conditional_achivments(gd, jnp.array(\n\t[\n\t\t1 if a in [Achievement.DEFEAT_ZOMBIE.value, Achievement.PLACE_TABLE.value] else 0\n\t\tfor a in range(Achievement.PLACE_TABLE.value + 1)\n\t]))'"
    }, 

        'INSTRUCTION_33':
        {
            'instruction': "Collect stones, place them and get a drink.",
            'instruction_paraphrases': 
            [
                "Gather rocks, set them down and grab a beverage.",
                "Acquire pebbles, position them accordingly and fetch a drink.",
                "Procure some cobblestones, arrange them suitably and have a refreshing drink.",
                "Amass a pile of stones, install them at an appropriate location and aid yourself to a drink.",
                "Assemble a collection of stone items, dispatch them in a certain configuration and partake in the enjoyment of a drink."
            ],
            'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
                                            1 if a in [Achievement.COLLECT_STONE.value, Achievement.PLACE_STONE.value, Achievement.COLLECT_DRINK.value] else 0
                                            for a in range(Achievement.MAKE_IRON_SWORD.value + 1)
                                        ])),
            'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_STONE.value, Achievement.PLACE_STONE.value, Achievement.COLLECT_DRINK.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value + 1)]))"
        }
   , 

        'INSTRUCTION_34': \
        {
            'instruction': "Start by placing a plant, then continue with the placement of the furnace.",
            'instruction_paraphrases': [
                "First, position the plant, then proceed to set the furnace.",
                "The plant is the first to be situated, and following that, make sure the furnace is put into place.",
                "Begin with the positioning of the flora, after which the heating device should be arranged.",
                "Take a plant and put it down first. After doing that, you should continue with placing the furnace.",
                "To start, place a plant and following that you should lay the furnace."
            ],
            'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
                1 if a in [Achievement.PLACE_PLANT.value, Achievement.PLACE_FURNACE.value] else 0
                for a in range(Achievement.MAKE_IRON_SWORD.value+1)
            ])),
            'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.PLACE_PLANT.value, Achievement.PLACE_FURNACE.value] else 0 for a in range(Achievement.PLACE_FURNACE.value+1)]))"
        }
   , 

        'INSTRUCTION_35':
        {
             'instruction': "First, collect some rocks. Then, make a stone sword.",
             'instruction_paraphrases': [
                   "Start by gathering some stones, after that, forge a sword from the stones you've gathered.",
                   "Your initial task is to amass a collection of rocks, using these rocks, proceed to create a stone sword.",
                   "The first course of action should involve you amassing an array of stones, then, employ these stones and manufacture a stone blade.",
                   "The foremost task you must undertake requires you to procure an assortment of geological rocks, next fabricate a weapon of stone.",
                   "Before anything else, you are to undertake the deposition of an accumulation of boulders, afterwards follow-through with the formation of a blade of stony constitution."
             ],
             'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
                 1 if a in [Achievement.COLLECT_STONE.value, Achievement.MAKE_STONE_SWORD.value] else 0
                 for a in range(Achievement.MAKE_IRON_SWORD.value+1)])),
             'str_check_lambda': "check_lambda: lambda gd, ix: conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_STONE.value, Achievement.MAKE_STONE_SWORD.value] else 0 for a in range(Achievement.MAKE_STONE_SWORD.value+1)]))"
        }
   , 

    'INSTRUCTION_36': {
        'instruction': "Gather some diamonds and make a stone pickaxe.",
        'instruction_paraphrases': [
            "Collect some diamonds and create a stone pickaxe.",
            "Acquire a few diamonds and craft a pickaxe made of stone.",
            "Procure a handful of diamonds and forge a stone pick.",
            "Amass a quantity of diamonds and fabricate a pick crafted from stones.",
            "Hoist in some diamond gems and construct a stone based pickaxe."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_DIAMOND.value, Achievement.MAKE_STONE_PICKAXE.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda':"""
            conditional_achivments(gd, jnp.array([
                1 if a in [Achievement.COLLECT_DIAMOND.value, Achievement.MAKE_STONE_PICKAXE.value] else 0
                for a in range(Achievement.MAKE_IRON_SWORD.value+1)
            ]))"""
    }, 

    'INSTRUCTION_37': {
        'instruction': "Craft a wooden pickaxe first, then start collecting some diamonds.",
        'instruction_paraphrases': [
            "Start by making a wooden pickaxe, then proceed to mining diamonds.",
            "Your first task is to create a wood miner's tool, once done move on to gather diamond gems.",
            "Creating a lumber pick is a top priority, afterwards, you can gather a few precious diamonds.",
            "Engage in crafting a timber pickaxe prior to embarking on your diamond collection endeavour.",
            "Commence with the construction of a wood-based pickaxe and later move towards the accumulation of several diamond stones."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.MAKE_WOOD_PICKAXE.value, Achievement.COLLECT_DIAMOND.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.MAKE_WOOD_PICKAXE.value, Achievement.COLLECT_DIAMOND.value] else 0 for a in range(Achievement.COLLECT_DIAMOND.value+1)]))"
    }, 

    'INSTRUCTION_38_1': {
        'instruction': "Please, gather some coal, create an iron pickaxe and eat a plant.",
        'instruction_paraphrases': [
            "Could you please mine some coal, forge an iron pickaxe and consume a plant?",
            "Collect a bit of coal, create a pickaxe made of iron and have a plant for food.",
            "Obtain some coal, manufacture an iron harvesting tool and eat greenery.",
            "Secure a quota of coal, assemble a miner's tool of iron and nourish yourself with a plant.",
            "Could you, kindly, procure a small batch of coal, put together a mineral extractor made of iron and ingest some vegetation?"
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_COAL.value, Achievement.MAKE_IRON_PICKAXE.value, Achievement.EAT_PLANT.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda':"conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_COAL.value, Achievement.MAKE_IRON_PICKAXE.value, Achievement.EAT_PLANT.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
    }, 

    'INSTRUCTION_39': \
      {
          'instruction': "You need to wake up, place your workbench and do not plant anything.",
          'instruction_paraphrases': [
              "Start by waking up, then set up your workbench, and remember not to plant anything.",
              "Before anything else, wake up, then place your workbench, yet avoid planting any seeds.",
              "Your tasks are to awaken, position your craft table, and abstain from any form of planting.",
              "The main actions you need to perform are rising from sleep, setting the crafting table in place, and steering clear of any kind of cultivation.",
              "You must rouse yourself from sleep, establish your workstation, and refrain from engaging in any horticultural activities."
          ],
          'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
              1 if a in [Achievement.WAKE_UP.value, Achievement.PLACE_TABLE.value] else
              -1 if a == Achievement.PLACE_PLANT.value else 0
              for a in range(Achievement.MAKE_IRON_SWORD.value+1)
          ])),
          'str_check_lambda':"conditional_achivments(gd, jnp.array([1 if a in [Achievement.WAKE_UP.value, Achievement.PLACE_TABLE.value] else -1 if a == Achievement.PLACE_PLANT.value else 0 for a in range(Achievement.PLACE_FURNACE.value+1)]))"
      }
 , 

        'GET_COAL': 
        {
            'instruction': "Collect some coal",
            'instruction_paraphrases': 
            [
                "Kindly gather some coal",
                "Can you procure some coal?",
                "I need you to harvest some coal",
                "Could you accumulate some black minerals for me?",
                "Ensure to have some coal in your inventory"
            ],
            'check_lambda':lambda gd, ix: conditional_achivments(gd, jnp.array([
                1 if a == Achievement.COLLECT_COAL.value else 0 
                for a in range(Achievement.MAKE_IRON_SWORD.value+1)
            ])),
            'str_check_lambda':"""conditional_achivments(gd, jnp.array([
                1 if a == Achievement.COLLECT_COAL.value else 0 
                for a in range(Achievement.MAKE_IRON_SWORD.value+1)
            ]))"""
        }
   , 

    'WAKE_UP': {
        'instruction': "Please wake up.",
        'instruction_paraphrases': [
            "Could you please awaken?",
            "I need you to rise and shine.",
            "It's time for you to wake up.",
            "You must awaken now.",
            "Can you rouse yourself from sleep?"
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.WAKE_UP.value else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': 'conditional_achivments(gd, jnp.array([1 if a == Achievement.WAKE_UP.value else 0 for a in range(Achievement.WAKE_UP.value+1)]))'
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
    }, 

    'INSTRUCTION_51': \
    {
        'instruction': "Collect a diamond and create a stone pickaxe.",
        'instruction_paraphrases': [
            "Find and gather a precious gem, then make a pickaxe out of stone.",
            "Initiate the collection of a diamond and proceed to fabricate a pickaxe using rock.",
            "I need you to acquire a diamond first, and afterwards construct a stone miner's tool.",
            "Could you please find a sparkling gem, and then make a tool for mining, specifically a pickaxe, with some stone?",
            "Would you be so kind as to secure a diamond for us, and following that, manufacture a mining instrument out of boulder, a pickaxe to be precise?",
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_DIAMOND.value, Achievement.MAKE_STONE_PICKAXE.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': "conditional_achivments(...)"
    }, 

    'INSTRUCTION_52 1':
    {
        'instruction': "Gather some wood and create a Stone Sword.",
        'instruction_paraphrases': 
        [
            "Collect timber and craft a blade of stone.",
            "Accumulate logs and manufacture a stone saber.",
            "Harvest wood and forge a sword from stone.",
            "Amass some lumber and make a weapon out of stone.",
            "Hoard some wooden materials and produce an weapon made of hard rock."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array(
        [
            1 if a in [Achievement.COLLECT_WOOD.value, Achievement.MAKE_STONE_SWORD.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_WOOD.value, Achievement.MAKE_STONE_SWORD.value] else 0 for a in range(Achievement.MAKE_STONE_SWORD.value+1)]))"
    },

    'INSTRUCTION_53': {
        'instruction': "Place a stone, eat a plant, and defeat a skeleton.",
        'instruction_paraphrases': [
            "Put down a rock, consume a herb, and conquer a skeleton.",
            "Set a stone in place, feed on a vegetable, and overcome a skeleton.", 
            "Lay a stone, ingest a plant, and vanquish a skeleton.",
            "Position a rock in an appropriate location, eat a piece of vegetation, and triumph over a skeleton.",
            "Establish a boulder in your surroundings, consume some verdure, and emerge victorious against a bony nemesis."
        ],
        'check_lambda': lambda gd, ix: 
conditional_achivments(gd, jnp.array(
            [1 if a in [Achievement.PLACE_STONE.value, Achievement.EAT_PLANT.value, Achievement.DEFEAT_SKELETON.value] else 0 
             for a in range(Achievement.MAKE_IRON_SWORD.value+1)]
        )),
        'str_check_lambda': 
"conditional_achivments(gd, jnp.array([1 if a in [Achievement.PLACE_STONE.value, Achievement.EAT_PLANT.value, Achievement.DEFEAT_SKELETON.value] else 0 for a in range(Achievement.DEFEAT_SKELETON.value+1)]))"
        }, 
'INSTRUCTION_54': \
        {'instruction': "Please gather drink but do not put down a furnace.",
        'instruction_paraphrases': [
            "Kindly collect liquid refreshments but avoid setting up any furnace.",
            "We need you to pick up some beverages but refrain from placing any sort of furnace.",
            "Could you collect some drinks? However, please do not position a furnace anywhere.",
            "You need to accumulate drink items but please make sure not to install a furnace.",
            "Your mission is to obtain liquids while ensuring that no furnace is being set up."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.COLLECT_DRINK.value else
            -1 if a == Achievement.PLACE_FURNACE.value else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value + 1) 
        ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a == Achievement.COLLECT_DRINK.value else -1 if a == Achievement.PLACE_FURNACE.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
    }, 

        'DEFEAT_ZOMBIE_AND_MAKE_IRON_SWORD': 
        {
            'instruction': "Defeat the zombie and make an iron sword.",
            'instruction_paraphrases': 
            [
                "Vanquish the undead and forge a sword from iron.", 
                "Overcome a zombie and fashion an iron blade.", 
                "Conquer a living dead and shape a blade out of iron.", 
                "Neutralize a zombie and create a weapon made from iron.", 
                "Overthrow a risen corpse and build an iron cutting weapon."
            ],
            'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
                1 if a in [Achievement.DEFEAT_ZOMBIE.value, Achievement.MAKE_IRON_SWORD.value] else
                0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
            ])),
            'str_check_lambda': 'conditional_achivments(gd, jnp.array([1 if a in [Achievement.DEFEAT_ZOMBIE.value, Achievement.MAKE_IRON_SWORD.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))'
        }
   , 

    'INSTRUCTION_57': {
        'instruction': "Plant an object and defeat a zombie",
        'instruction_paraphrases': [
            "Sow a plant and destroy a zombie",
            "Cultivate a seedling and take down a walker",
            "Establish a vegetation and conquer an undead",
            "Establish a greenery and vanquish a deadhead",
            "Initiate a horticulture and neutralize a living dead"
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.PLACE_PLANT.value, Achievement.DEFEAT_ZOMBIE.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.PLACE_PLANT.value, Achievement.DEFEAT_ZOMBIE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
    }, 

    'INSTRUCTION_58_1': {
    
        'instruction': "Beat the skeleton but don't plant anything and gather wood.",
        'instruction_paraphrases': [
            "Vanquish the bony monster, eschew planting and obtain timber.",
            "Wood collecting is essential, defeat the skeleton and abstain from planting anything.",
            "It's paramount you defeat the skeleton and collect some wood, however make sure not to plant anything.",
            "Ensuring no plantation is done, gather logs and prevail over the skeleton.",
            "Acquire wood, victorious be against the skeleton whilst abstaining from any form of cultivation."
        ],
        
        'check_lambda': lambda gd, ix:conditional_achivments(gd, jnp.array([1 if a in [Achievement.DEFEAT_SKELETON.value, Achievement.COLLECT_WOOD.value] else -1 if a == Achievement.PLACE_PLANT.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.DEFEAT_SKELETON.value, Achievement.COLLECT_WOOD.value] else -1 if a == Achievement.PLACE_PLANT.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
   
   }, 
'INSTRUCTION_59': \
        {'instruction': "Defeat a zombie, place a stone and make a wooden sword.",
        'instruction_paraphrases': [
            "Bring down a creeper, deploy a rock, and craft a timber blade.",
            "Beat an undead, set down a stone, and create a wooden cutter.",
            "Subdue a ghoul, position a boulder, and manufacture a lumber sword.",
            "Overcome a zombie, install a solid stone and produce a sword out of wood.",
            "Triumph over a zombie, arrange a cobble stone and shape a sword with the help of wood."],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.DEFEAT_ZOMBIE.value, Achievement.PLACE_STONE.value, Achievement.MAKE_WOOD_SWORD.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.DEFEAT_ZOMBIE.value, Achievement.PLACE_STONE.value, Achievement.MAKE_WOOD_SWORD.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
    }, 

    'INSTRUCTION_611': {
        'instruction': "Collect diamonds, place a plant and gather wood.",
        'instruction_paraphrases': [
            "Acquire some precious gemstones, establish greenery, and obtain timber.",
            "Secure some diamonds, deposit a flower, and harvest wood.",
            "Procure jewels, plant a seedling and collect some logs.",
            "Obtain shiny crystals, position a flora, and gather some hardwood.",
            "Grasp some dazzling rocks, situate a vegetation, and accumulate wooden materials."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_DIAMOND.value, Achievement.PLACE_PLANT.value, Achievement.COLLECT_WOOD.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_DIAMOND.value, Achievement.PLACE_PLANT.value, Achievement.COLLECT_WOOD.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))",
    }, 

    'INSTRUCTION_62': {
        'instruction': "Forge an iron pick, avoid obtaining any drinks at all, and make a wooden sword.",
        'instruction_paraphrases': [
            "Fashion a pickaxe from iron, don't collect any beverages, and create a sword out of wood.",
            "Build a wooden sword, abstain from getting any sort of drink, and also craft an iron pickaxe.",
            "Craft a timber blade, whilst ensuring you do not gather any drinks, also construct an iron gut hook.",
            "Avoid grabbing any liquid refreshments and concentrate on constructing iron diggers and wooden sabers.",
            "Establish your inventory with the likes of an iron dull miner's tool and a sword made from forest produce but steer clear from hydration refreshments of any kind."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([ 
            1 if a in [Achievement.MAKE_IRON_PICKAXE.value, Achievement.MAKE_WOOD_SWORD.value] else 
            -1 if a == Achievement.COLLECT_DRINK.value else 0 
            for a in range(Achievement.MAKE_IRON_SWORD.value+1) 
        ])),
        
        'str_check_lambda': "conditional_achivments(gd, jnp.array([ 1 if a in [Achievement.MAKE_IRON_PICKAXE.value, Achievement.MAKE_WOOD_SWORD.value] else -1 if a == Achievement.COLLECT_DRINK.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1) ]))" 
    }, 

    'INSTRUCTION_63': {
        'instruction': "Please make sure you do not craft a wooden sword. But, you should place a furnace.",
        'instruction_paraphrases': [
            "Kindly avoid creating a wooden blade. However, you should set up an oven.",
            "Don't even think about constructing a wood-crafted saber. Instead, I need you to put up a stove.",
            "Steer clear of crafting a wooden sword. But, please go ahead and place a kiln.",
            "Under no circumstances should you produce a wood-based sword. In contrast, you are expected to install a heating appliance.",
            "The crafting of the wooden broadsword is not to be endeavoured. In lieu of that, establish a hearth's presence."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.PLACE_FURNACE.value] else
            -1 if a == Achievement.MAKE_WOOD_SWORD.value else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': """conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.PLACE_FURNACE.value] else
            -1 if a == Achievement.MAKE_WOOD_SWORD.value else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))"""
    }, 

    'INSTRUCTION_64': \
        {
        'instruction': "Put a stone, wake up and collect some iron.",
        'instruction_paraphrases': [
            "Wake up, and then proceed to collect iron after setting a rock down.",
            "You need to place a rock down, wake yourself up and then head out to gather some iron.",
            "Start by placing a boulder, make sure to be awake, and finish by gathering iron.",
            "You must ensure to rouse yourself, carefully place down a stone, and then engage in the task of iron collection.",
            "Activate your wakefulness, go through the process of positioning a stone, and post this, industriously collate iron."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.PLACE_STONE.value, Achievement.WAKE_UP.value, Achievement.COLLECT_IRON.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda':"conditional_achivments(gd, jnp.array([1 if a in [Achievement.PLACE_STONE.value, Achievement.WAKE_UP.value, Achievement.COLLECT_IRON.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))" 
        }
   , 

        'PLACE_PLANT_AND_WAKE_UP': 
        {
            'instruction': "Place a plant and wake up",
            'instruction_paraphrases': 
            [
                "Set down a seedling and then rouse from sleep",
                "Put a sapling in place and awaken",
                "Plant a seed and stir from slumber",
                "Make sure a sprout is in its spot before breaking from your repose",
                "Ensure that a shoot is correctly positioned, afterwards, cease your rest"
            ],
            'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array
                (
                    [
                        1 if a in [Achievement.PLACE_PLANT.value, Achievement.WAKE_UP.value] else 0 
                        for a in range(Achievement.MAKE_IRON_SWORD.value+1)
                    ]
                )),
            'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.PLACE_PLANT.value, Achievement.WAKE_UP.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
        }
   , 

    'INSTRUCTION_66': {
        'instruction': "Make a wooden pickaxe, slay a skeleton with a wooden sword, and defeat a skeleton.",
        'instruction_paraphrases': [
            "Create a lumber pick, combat a skeleton with a timber blade, and vanquish a skeleton.",
            "Manufacture a wood pickaxe, engage with a skeleton using a wood rapier, and conquer a skeleton.",
            "Forge a wooden digging tool, clash with a bony antagonist using a wooden edge weapon, and be victorious against a skeletal enemy.",
            "Fabricate a wooden mining accessory, wage a battle against a hard tissue villain with a wooden Sabre, and be triumphant against a calcium-rich framework adversary.",
            "Construct a forestry mineral extraction tool, exercise a skirmish with an osseous foe with an arboreal broadsword, and emerge victorious against a skeletal adversary.",
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.MAKE_WOOD_PICKAXE.value, Achievement.MAKE_WOOD_SWORD.value, Achievement.DEFEAT_SKELETON.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([ 1 if a in [Achievement.MAKE_WOOD_PICKAXE.value, Achievement.MAKE_WOOD_SWORD.value, Achievement.DEFEAT_SKELETON.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
    }, 

    'STONE_PICKAXE_COW_PLANT': {
        'instruction': "Make a stone pickaxe, eat a cow and eat a plant.",
        'instruction_paraphrases': [
            "Forge a stone cutter, feast on a cow, and have a plant for refreshment.",
            "Create a cutter made of rock, a cow is to be your nourishment along with gorging on a plant.",
            "Shape a chisel from the hardened earth, consume a bovine creature and chew on a portion of flora.",
            "Transform a piece of bedrock into a tool for breaking apart the ground, quench your hunger by the flesh of a grazing animal and a piece of greenery.",
            "Carve a hardened mineral into a primitive excavation instrument, ingest the meat of a domesticated mammal and satiate your thirst with a botanical specimen."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.MAKE_STONE_PICKAXE.value, Achievement.EAT_COW.value, Achievement.EAT_PLANT.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': ("conditional_achivments(..., jnp.array(["
                             "1 if a in [Achievement.MAKE_STONE_PICKAXE.value, Achievement.EAT_COW.value, "
                             "Achievement.EAT_PLANT.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)])")
    }, 

    'INSTRUCTION_68': {
        'instruction': "Collect diamonds and craft a stone sword.",
        'instruction_paraphrases': [
            "Find some diamonds and make a stone blade.",
            "Hunt for diamonds and forge a sword from stone.",
            "Dig up diamonds and fabricate a sword out of rock.",
            "Search for diamonds and construct a stone-cutting weapon.",
            "Spend time in mining diamonds, afterwards use the stone material to create a combat sword."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_DIAMOND.value, Achievement.MAKE_STONE_SWORD.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda':"conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_DIAMOND.value, Achievement.MAKE_STONE_SWORD.value] else 0 for a in range(Achievement.MAKE_STONE_SWORD.value+1)]))"
    }, 

    'PLACE_STONE_TABLE_NOT_PLANT': {
        'instruction': "Set down the stone and the table, but do not place the plant.",
        'instruction_paraphrases': [
            "Please put the rock and the workbench down, but refrain from placing the seedling.",
            "Kindly position the pebble and the desk but avoid positioning the sapling.",
            "You should arrange the boulder and the countertop but not the shrub.",
            "It's required to locate the cobble and the tabletop but shun locating the small tree.",
            "It's vital to establish the position of the grindstone and the trestle table but avoid the sprig."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.PLACE_STONE.value, Achievement.PLACE_TABLE.value] else
            -1 if a == Achievement.PLACE_PLANT.value else 0 
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': "lambda gd, ix: conditional_achivments(gd, jnp.array([1 if a in [Achievement.PLACE_STONE.value, Achievement.PLACE_TABLE.value] else -1 if a == Achievement.PLACE_PLANT.value else 0 for a in range(Achievement.PLACE_STONE.value+1)]))"
    }, 

        'INSTRUCTION_70': 
            {
            'instruction': "Make sure to place a crafting table but avoid planting anything",
            'instruction_paraphrases': [
                "Set up the crafting table but do not put down any plants",
                "The workbench needs to be set up, but do not sow any saplings",
                "Put the crafting station in place, but stay away from planting",
                "Install the crafting bench but stay clear of the act of sowing",
                "Arrange the work station but avoid the act of cultivation"
                ],
            'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
                    1 if a == Achievement.PLACE_TABLE.value else
                    -1 if a == Achievement.PLACE_PLANT.value else 0
                    for a in range(Achievement.MAKE_IRON_SWORD.value+1)
                ])),
            'str_check_lambda':"conditional_achivments(gd, jnp.array([ 1 if a == Achievement.PLACE_TABLE.value else -1 if a == Achievement.PLACE_PLANT.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1) ]))"
            }
   , 

    'INSTRUCTION_71': {
        'instruction': "Please gather wood, defeat a skeleton and collect a sapling.",
        'instruction_paraphrases': [
            "Could you please collect timber, slay a skeleton and pick up a tree sapling?",
            "I would appreciate if you can procure lumber, vanquish a skeleton and gather a seedling.",
            "Can you bring me some firewood, overcome a bony adversary, and pick up a small tree shoot?",
            "It is required for you to harvest tree materials, vanquish a bony combatant, and accumulate a young tree.",
            "It is necessary for you to obtain log, overpower a skeleton, and secure a small plant originated from a seed.",
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_WOOD.value, Achievement.DEFEAT_SKELETON.value, Achievement.COLLECT_SAPLING.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': """conditional_achivments(gd, jnp.array([
        1 if a in [Achievement.COLLECT_WOOD.value, Achievement.DEFEAT_SKELETON.value, Achievement.COLLECT_SAPLING.value] else 0
        for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"""
    }, 

        'INSTRUCTION_72': 
        {
            'instruction': "Gather some timber but avoid planting anything.",
            'instruction_paraphrases': 
            [
                "Collect wood pieces but refrain from placing any plants.",
                "Pool some woodlogs. Make sure not to place any saplings.",
                "Assemble a good amount of lumber but don't get involved with sapling placement.",
                "Procure some wooden blocks and steer clear from putting any sprouts into the ground.",
                "Accumulate chunks of wood, yet do not commit to the act of setting down any seedlings."
            ],
            'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array(
            [
                1 if a == Achievement.COLLECT_WOOD.value else
                -1 if a == Achievement.PLACE_PLANT.value else 0
                for a in range(Achievement.MAKE_IRON_SWORD.value+1)
            ])),
            'str_check_lambda': "conditional_achivments(gd, jnp.array([ \
            1 if a == Achievement.COLLECT_WOOD.value else \
            -1 if a == Achievement.PLACE_PLANT.value else 0 \
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
        }
   , 

        'INSTRUCTION_731': 
        {
            'instruction': "Collect some diamond and place a furnace.",
            'instruction_paraphrases': [
                "Find some diamond and setup a furnace.",
                "Grab some diamond then put a furnace.",
                "Acquire diamond and situate a furnace",
                "Obtain diamond and establish a furnace."
            ],
            'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
                1 if a in [Achievement.COLLECT_DIAMOND.value, Achievement.PLACE_FURNACE.value] else 0 
                for a in range(Achievement.MAKE_IRON_SWORD.value+1)
            ])),
            'str_check_lambda':"conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_DIAMOND.value, Achievement.PLACE_FURNACE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
        }
   , 

    'MAKE_STONE_PICKAXE AND EAT_COW AND PLACE_FURNACE': {
        'instruction': "Craft a stone pickaxe, consume some beef and set up a furnace.",
        'instruction_paraphrases': [
            "Use some rocks to make a pickaxe, eat a piece of cow meat and establish a furnace.",
            "Create a pickaxe from the stones, feast on some cow and erect a furnace.",
            "Fashion a pickaxe using stones, devour a slice of beef and install a cooking furnace.",
            "Formulate a stone-based pickaxe, consume a portion of bovine sustenance and position a furnace in place.",
            "Construct a pickaxe through the usage of stones, partake in the intake of beef from a cow and implement the installation of a furnace."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.MAKE_STONE_PICKAXE.value, Achievement.EAT_COW.value, Achievement.PLACE_FURNACE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
            ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.MAKE_STONE_PICKAXE.value, Achievement.EAT_COW.value, Achievement.PLACE_FURNACE.value] else 0 for a in range(Achievement.PLACE_FURNACE.value+1)]))"
    }, 

    'INSTRUCTION_75': {
        'instruction': "Make sure you have placed a table, collected a sapling, and gathered a drink.",
        'instruction_paraphrases': [
            "Ensure that you have set up a workbench, picked up a sprout, and acquired a beverage.",
            "It's imperative that you lay down a desk, garner a tree seedling and fetch a liquid refreshment.",
            "Ascertain that you have installed a counter, harvested a young tree, and secured a thirst quencher.",
            "Be certain that you have situated a bench, accumulated a sapling, and collected a libation.",
            "Verify that you have positioned a tabletop, reaped a seedling, and amassed a drink."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            
            1 if a in [Achievement.PLACE_TABLE.value, Achievement.COLLECT_SAPLING.value, Achievement.COLLECT_DRINK.value] else 0 
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        
        ])),
        'str_check_lambda':"""conditional_achivments(gd, jnp.array([
            
            1 if a in [Achievement.PLACE_TABLE.value, Achievement.COLLECT_SAPLING.value, Achievement.COLLECT_DRINK.value] else 0 
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        
        ]))"""
                        }, 

    'INSTRUCTION_76':
      {
          'instruction': "You need to gather some lumber and consume a plant, just make sure you don't plant anything.",
          'instruction_paraphrases': [
              "Collect some wood and eat a plant, but refrain from planting any seeds.",
              "Work on obtaining timber and remember to eat a plant, however, avoid engaging in any form of planting.",
              "Have the task of securing logs and consuming some greens, but do not engage in any planting activities.",
              "Allocate time to accumulate wooden materials and feast on a plant, yet it is essential not to sow any type of seed.",
              "As an integral part of your mission, the acquisition of firewood and intake of a plant are essential, while deliberately avoiding the planting of seeds."
          ],
          'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
                1 if a in [Achievement.COLLECT_WOOD.value, Achievement.EAT_PLANT.value] else
                -1 if a == Achievement.PLACE_PLANT.value else 0
                for a in range(Achievement.MAKE_IRON_SWORD.value+1)
          ])),
          'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_WOOD.value, Achievement.EAT_PLANT.value] else -1 if a == Achievement.PLACE_PLANT.value else 0 for a in range(Achievement.PLACE_FURNACE.value+1)]))"
      }
 , 

    'INSTRUCTION_77': {
        'instruction': "Collect a drink, gather a sapling and then place a plant.",
        'instruction_paraphrases': [
            "Grab a beverage, obtain a small tree and put a plant down.",
            "Gather a refreshment, pick up a seedling and set a vegetation.",
            "Procure a liquid refreshment, collect a young tree and position a flora.",
            "Fetch a drinkable, accumulate a tree bud and place a greenery.",
            "Gain a potable, secure a young plant and station a herb."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_DRINK.value, Achievement.COLLECT_SAPLING.value, Achievement.PLACE_PLANT.value] else 0 
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_DRINK.value, Achievement.COLLECT_SAPLING.value, Achievement.PLACE_PLANT.value] else 0 for a in range(Achievement.PLACE_PLANT.value+1)]))"
    }, 

    'INSTRUCTION_78': {
        'instruction': "Place a stone, eat a plant, and collect some coal.",
        'instruction_paraphrases': [
            "Lay down a rock, munch on some greenery, and gather a bit of charcoal.",
            "Stick a boulder somewhere, consume some vegetation and stock up on coal.",
            "Position a stone properly, feast on a plant, and end by mining some coal.",
            "Put a piece of rock in a desired location, have a plant for your meal, and make sure to accumulate some coal.",
            "Establish the location of a chunk of stone, indulge in consuming some flora, and don't forget to amass an amount of carbon."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.PLACE_STONE.value, Achievement.EAT_PLANT.value, Achievement.COLLECT_COAL.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.PLACE_STONE.value, Achievement.EAT_PLANT.value, Achievement.COLLECT_COAL.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
    }, 

    'INSTRUCTION_79': {
        'instruction': "Don't craft a wooden sword but make sure to collect saplings twice.",
        'instruction_paraphrases': [
            "Ensure you collect saplings two times, but do not create a wooden sword.",
            "Gather saplings on two occasions and abstain from constructing a wooden weapon.",
            "Twice, you must collect plant shoots, but constructing a timber sword is prohibited.",
            "Although wooden swords are not to be fashioned, it's imperative that saplings are gathered in two instances.",
            "The manufacturing of wooden weaponry is disallowed, however, a dual procurement of saplings is required."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            -1 if a == Achievement.MAKE_WOOD_SWORD.value else
            1 if a == Achievement.COLLECT_SAPLING.value else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([-1 if a == Achievement.MAKE_WOOD_SWORD.value else 1 if a == Achievement.COLLECT_SAPLING.value else 0 for a in range(Achievement.MAKE_WOOD_SWORD.value+1)]))"
    }, 

    'COLLECT_SAPLING_MAKE_WOOD_PICKAXE_MAKE_IRON_SWORD': {
        'instruction': "First, accumulate the sapling. Then, craft a wooden pickaxe. Finally, forge an iron sword.",
        'instruction_paraphrases': [
            "Initially, gather the sapling, fabricate a pickaxe from wood and eventually, form an iron sword.",
            "Start by concentrating on the sapling collection. Afterwards, create a pickaxe using wood. At last, utilize iron to construct a sword.",
            "First of all, work on obtaining a sapling. Next on the agenda is the production of a wooden pickaxe. Finally culminate your tasks with the creation of a sword forged from iron.",
            "The first item on your list should be the collection of a sapling. Move on to creating a handy tool, the wooden pickaxe. Wrap up your tasks with the intricate process of forging a sword made of metal - specifically, iron.",
            "Kick off your game with the essential task of procuring a sapling. Use your crafting skills to give life to a wooden pickaxe. Lastly, leave your mark by crafting an iron sword."
        ],
        'check_lambda': lambda gd, ix : conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_SAPLING.value, Achievement.MAKE_WOOD_PICKAXE.value, Achievement.MAKE_IRON_SWORD.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value + 1)
        ])),
        'str_check_lambda': '''conditional_achivments(gd,
            jnp.array([ 1 if a in [Achievement.COLLECT_SAPLING.value, Achievement.MAKE_WOOD_PICKAXE.value, Achievement.MAKE_IRON_SWORD.value] 
                else 0 for a in range(Achievement.MAKE_IRON_SWORD.value + 1)]))
        '''}, 

    'INSTRUCTION_82': 
    {
        'instruction': "Collect a sapling and then eat a cow.",
        'instruction_paraphrases': [
            "Get hold of a seedling first, later proceed to consume a cow.",
            "Grab a young plant, after doing so, feed on a bovine.",
            "Pick a sprout and subsequently dine on beef.",
            "First, lay your hands on a small tree. Once that task is done, eat some cow meat.",
            "Pursuit the acquisition of a sprinkling and in the sequence partake in the act of eating a ruminant creature."      
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array(
            [
                1 if a in [Achievement.COLLECT_SAPLING.value, Achievement.EAT_COW.value] else 0
                for a in range(Achievement.MAKE_IRON_SWORD.value+1)
            ]
        )),
        'str_check_lambda':"conditional_achivments(...)"
    } , 

    'DO_NOT_MAKE_STONE_SWORD_AND_COLLECT_DIAMOND_AND_EAT_PLANT': {
        'instruction': "Avoid crafting a stone sword, locate and pick up a diamond, and make sure to consume a plant.",
        'instruction_paraphrases': [
            "Steer clear of making a rock blade, get hold of a precious stone, and don't forget to eat some vegetation.",
            "Keep away from fabricating a cobblestone cutter, seek and obtain a dazzling gem, and chomp on a piece of flora.",
            "Bypass the creation of an ore slicer, unearth a glittering jewel, and indulge in some herbal nourishment.",
            "Eschew the construction of a mineral cleaver, discover and acquire a sparkling gemstone, and partake in botanical consumption."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd,jnp.array([ 
            -1 if a == Achievement.MAKE_STONE_SWORD.value else
            1 if a in [Achievement.COLLECT_DIAMOND.value,Achievement.EAT_PLANT.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': 'conditional_achivments(gd,jnp.array([ -1 if a == Achievement.MAKE_STONE_SWORD.value else 1 if a in [Achievement.COLLECT_DIAMOND.value,Achievement.EAT_PLANT.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1) ]))'
    }, 

    'INSTRUCTION_84': \
    {
        'instruction': "Make an iron sword, ensure you have not made a wooden sword and place a furnace.",
        'instruction_paraphrases': [
            "Construct an iron blade, don't build a wooden blade and make sure to install a furnace.",
            "Create an iron sword while avoiding creating a wooden sword and set up a furnace.",
            "Place a furnace and craft an iron sword but refrain from making a wooden sword.",
            "Forge a sword from iron, place down a furnace and make sure not to make a wooden weapon",
            "Affix a furnace, assemble a sword using iron and don't use wood to form a sword."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.PLACE_FURNACE.value, Achievement.MAKE_IRON_SWORD.value] else
            -1 if a == Achievement.MAKE_WOOD_SWORD.value else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda':"conditional_achivments(gd, jnp.array([1 if a in [Achievement.PLACE_FURNACE.value, Achievement.MAKE_IRON_SWORD.value] else -1 if a == Achievement.MAKE_WOOD_SWORD.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))" 
    }, 

    'INSTRUCTION_87': {
        'instruction': "Collect a sapling, eat a cow, and make an iron sword.",
        'instruction_paraphrases': [
            "Make sure to gather a tree sprout, consume cow meat, and forge an iron blade.",
            "You must pick up a young tree, devour a beef, and construct a sword made of iron.",
            "It's essential that you procure a seedling, feast on a bovine, and fabricate a weapon from iron.",
            "It's necessary for you to accumulate a small tree, partake in a serving of cow, and synthesize a bladed weapon utilizing iron.",
            "The tasks you must undertake include acquiring a juvenile tree, eating meat sourced from a cow, and manufacturing a sharp iron instrument."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_SAPLING.value, Achievement.EAT_COW.value, Achievement.MAKE_IRON_SWORD.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_SAPLING.value, Achievement.EAT_COW.value, Achievement.MAKE_IRON_SWORD.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
    }, 

    'INSTRUCTION_88': {
        'instruction': "Ensure that you have collected a diamond and defeated a zombie.",
        'instruction_paraphrases': [
            "Make sure you've managed to pick up a diamond and successfully defeat a zombie.",
            "Verify that in your collection you have a precious gem and also defeated a walking dead.",
            "Confirm if you have collected a shining jewel and successfully battled a living dead.",
            "It is crucial that you've procured a sparkling precious stone and protected yourself by defeating a grotesque creature.",
            "As a mandate, please confirm if you've procured a shimmering diamond and successfully overcame the challenge of a scary undead creature."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.COLLECT_DIAMOND.value, Achievement.DEFEAT_ZOMBIE.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value + 1)
        ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_DIAMOND.value, Achievement.DEFEAT_ZOMBIE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value + 1)]))"
    }, 

'INSTRUCTION_89_1': 
    {
        'instruction': "Ensure you have defeated the skeleton and consumed the cow.",
        'instruction_paraphrases': [
            "Make sure you've battled the skeleton and eaten the cow.",
            "Confirm that you've vanquished the skeleton and have eaten the cow.",
            "Did you fight off the skeleton and eat the cow? You must do this!",
            "You need to have engaged with a skeleton in battle and have consumed a cow.",
            "It's necessary that you've dealt with the skeleton by defeating it and you've used a cow for sustenance."
        ],
        
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.DEFEAT_SKELETON.value, Achievement.EAT_COW.value] else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        
        'str_check_lambda':"conditional_achivments(gd, jnp.array([1 if a in [Achievement.DEFEAT_SKELETON.value, Achievement.EAT_COW.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
    }, 

    'INSTRUCTION_90 1': {
        'instruction': "Defeat a zombie, get some coal and put down a stone",
        'instruction_paraphrases': [
            "Beat up a zombie, procure some blackrock, then deposit a cobble",
            "Strike down one of the living dead, gather a piece of carbon and position a block of rock",
            "Knock down one of those creatures of the night, extract dark gemstone, then place a fragment of boulder",
            "Overwhelm the creature back from death, accumulate some fossil fuel, then prompt a unit of sediment",
            "Bring the restless monster to rest, collect black gold and position a portion of monolith"
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.DEFEAT_ZOMBIE.value, Achievement.COLLECT_COAL.value, Achievement.PLACE_STONE.value]
            else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda':"""lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.DEFEAT_ZOMBIE.value, Achievement.COLLECT_COAL.value, Achievement.PLACE_STONE.value]
            else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ]))"""
    }, 

    'INSTRUCTION_91': {
        'instruction': "Construct a stone pickaxe and a stone sword, but do not set up a furnace.",
        'instruction_paraphrases': [
            "Put together a stone blade and a stone axe handle, refrain from installing any oven.",
            "For this task, I want you to build a stone sword, create a stone pick-axe, but avoid establishing a furnace.", 
            "Eschew the assembly of a furnace, but prioritize the creation of both a sword and a pickaxe crafted from stone.",
            "Execute a construction of both a stone-laden sword and a stone pickaxe, however nullify any actions towards a furnace's installation.",
            "Forge ahead with assembling a melee weapon and a mining tool from stone, but abstain from setting up any kind of heat source." 
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a == Achievement.MAKE_STONE_PICKAXE.value or a == Achievement.MAKE_STONE_SWORD.value else
            -1 if a == Achievement.PLACE_FURNACE.value else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a == Achievement.MAKE_STONE_PICKAXE.value or a == Achievement.MAKE_STONE_SWORD.value else -1 if a == Achievement.PLACE_FURNACE.value else 0 for a in range(Achievement.PLACE_FURNACE.value+1)]))"
    }, 

        'INSTRUCTION_92_1': {
            'instruction': "Collect a drink and fight off a zombie.",
            'instruction_paraphrases': [
                "You need to gather a beverage and defeat a zombie.",
                "Acquire a refreshment and combat a living-dead.",
                "The task involves procuring a potion and battling a walker.",
                "Retrieve an elixir and launch an attack against the undead.",
                "Achieve the goal of collecting a draught and waging war against a zombie."
            ],
            'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
                1 if a in [Achievement.COLLECT_DRINK.value, Achievement.DEFEAT_ZOMBIE.value] else 0
                for a in range(Achievement.MAKE_IRON_SWORD.value+1)
            ])),
            'str_check_lambda':"conditional_achivments(gd, jnp.array([1 if a in [Achievement.COLLECT_DRINK.value, Achievement.DEFEAT_ZOMBIE.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))"
        }
   , 
\
    'INSTRUCTION_930': {
        'instruction': "Ensure that you do not place the furnace but create a stone pickaxe.",
        'instruction_paraphrases': [
            "Avoid setting up the furnace and instead make a stone pickaxe.",
            "Bypass arranging the heater, and bring into being a stone pick.",
            "Keep from situating the kiln but produce stone axe intended for mining.",
            "Steer clear of locating the coke oven but fabricate rock chopper.",
            "Evade standing the cooking range but contrive cobblestone mining tool."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            -1 if a in [Achievement.PLACE_FURNACE.value] else
            1 if a == Achievement.MAKE_STONE_PICKAXE.value else 0
            for a in range(Achievement.MAKE_IRON_SWORD.value+1)
        ])),
        'str_check_lambda': 'conditional_achivments'
    }, 

    'DEFEAT_SKELETON_AND_PLACE_PLANT': {
        'instruction': "Defeat a skeleton and then place a plant.",
        'instruction_paraphrases': [
            "Ensure to defeat a skeleton first, then after you can place a plant.",
            "Please take down a skeleton and plant a sapling.",
            "Eliminate a skeleton, following with the placement of a plant.",
            "Your mission is to neutralize a skeleton entity and proceed to plant a flora.",
            "Your objective is to overcome a skeletal combatant, subsequent to which you are to establish a botanical lifeform."
        ],
        'check_lambda': lambda gd, ix: conditional_achivments(gd, jnp.array([
            1 if a in [Achievement.DEFEAT_SKELETON.value, Achievement.PLACE_PLANT.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value + 1)
        ])),
        'str_check_lambda': "conditional_achivments(gd, jnp.array([1 if a in [Achievement.DEFEAT_SKELETON.value, Achievement.PLACE_PLANT.value] else 0 for a in range(Achievement.MAKE_IRON_SWORD.value + 1)]))"
    }
}

from craftext.scenarios.parce_dataset import update_previous_dict
from craftext.scenarios.constants import base_path
import os

medium_test_other_paramets = {}
medium_test_other_paramets = update_previous_dict(
    medium_test_other_paramets, 
    os.path.join(base_path, "jax_conditional_achivments/instructions/test/medium/other_params"), 
    "achivments"
)

medium_test_parafrased = {}
medium_test_parafrased = update_previous_dict(
    medium_test_parafrased, 
    os.path.join(base_path, "jax_conditional_achivments/instructions/test/medium/paraphrases"), 
    "achivments"
)


if __name__ == "__main__":
    import json
    instructions = []
    for key in easy_test_other_paramets.keys():
        instructions.append(easy_test_other_paramets[key]['instruction'])
        instructions += easy_test_other_paramets[key]['instruction_paraphrases']
    with open("instructions_achivments_easy_test_other_paramets.json", "w", encoding="utf-8") as json_file:
        json.dump(instructions, json_file, ensure_ascii=False, indent=4)
        
    for key in easy_test_parafrased.keys():
        try:
            instructions.append(easy_test_parafrased[key]['instruction'])
            instructions += easy_test_parafrased[key]['instruction_paraphrases']
        except:
            print(key)
    with open("instructions_achivments_easy_test_parafrased.json", "w", encoding="utf-8") as json_file:
        json.dump(instructions, json_file, ensure_ascii=False, indent=4)