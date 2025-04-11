import jax.numpy as jnp
from jax import lax

from craftext.adapters.state_adapter import PlayerInventory
from craftext.checkers_jax.target_state import TargetState
def check_inventory(inventory: PlayerInventory, object_inventory_enum, count_to_collect: int):
    """
    Checks the amount of a specific item in the player's inventory using a switch-based approach.
    
    :param inventory: Player's inventory (PlayerInventory).
    :param object_inventory_enum: Enum corresponding to the inventory item.
    :param count_to_collect: Required amount of the item.
    :return: Boolean indicating if the required amount was collected.
    """
    
    def get_item(index, inventory):
        return lax.switch(index,
            [
                lambda: inventory.wood,
                lambda: inventory.stone,
                lambda: inventory.coal,
                lambda: inventory.iron,
                lambda: inventory.diamond,
                lambda: inventory.sapling,
                
                # In instructions, don't use pickaxes or swords. Instead, use a constant, such as diamond. These variables have different sizes in Craftax, so their usage leads to errors.
                lambda: inventory.diamond,
                lambda: inventory.diamond,
                lambda: inventory.diamond,
                lambda: inventory.diamond,
                lambda: inventory.diamond,
                lambda: inventory.diamond,
                
                # lambda: lax.switch(inventory.inventory, [lambda: inventory.pickaxe, lambda: inventory.wood_pickaxe]), 
                # lambda: lax.switch(inventory.inventory, [lambda: inventory.sword, lambda: inventory.stone_pickaxe]),
                # lambda: lax.switch(inventory.inventory, [lambda: inventory.bow,   lambda: inventory.iron_pickaxe]), 
                # lambda: lax.switch(inventory.inventory, [lambda: inventory.arrows, lambda: inventory.wood_sword]),
                # lambda: lax.switch(inventory.inventory, [lambda: inventory.armour, lambda: inventory.stone_sword]),
                # lambda: lax.switch(inventory.inventory, [lambda: inventory.torches, lambda: inventory.iron_sword]),
            
                lambda: inventory.ruby,
                lambda: inventory.sapphire,
                
                ####
                lambda: inventory.diamond, #lambda: inventory.potions
                lambda: inventory.diamond, #lambda: inventory.books
            ]
        )
        
    collected_count = get_item(object_inventory_enum, inventory)
    return collected_count >= count_to_collect


def check_map(game_map: jnp.ndarray, object_to_place: int, count_to_stand: int):
    """
    Checks if the required number of `object_to_place` has been placed on the map.
    
    :param game_map: The game map.
    :param object_to_place: The index of the object to check.
    :param count_to_stand: Required number of placed objects.
    :return: Boolean indicating if the required amount of objects were placed on the map.
    """
    placed_count = jnp.sum(game_map == object_to_place)
    return placed_count >= count_to_stand

def conditional_placing(gd, target_state: TargetState):
    """
    The function that checks if:
    1) The required number of `object_inventory` was collected in the previous 
       state.
    2) The required number was collected in the current state.
    3) The required number of `object_to_place` was placed on the map in the 
       current state.
    
    :param gd: Game state history (GameDataClassic).
    :param object_inventory_enum: Numeric value corresponding to an inventory 
       item. Possible values correspond to items in the inventory: 
       WOOD (simple), STONE (simple), COAL (simple), 
       IRON (simple), DIAMOND (medium),SAPLING (simple), 
       WOOD_PICKAXE (simple), STONE_PICKAXE (simple), 
       IRON_PICKAXE (medium), WOOD_SWORD (simple), 
       STONE_SWORD (simple), IRON_SWORD (medium).
       
    :param object_to_place: Index of the object on the map. Possible values for 
       blocks that can be placed: "STONE"(simple), "CRAFTING_TABLE"(simple), "FURNACE"(simple), "CHEST"(medium), 
       "FOUNTAIN"(medium), "ENCHANTMENT_TABLE_FIRE"(medium), "ENCHANTMENT_TABLE_ICE"(medium), "PLANT"(simple).
       
    :param count_to_collect: Required number of objects in the inventory.
    :param count_to_stand: Required number of objects placed on the map.
    :return: Returns True if both conditions are satisfied in sequence, 
             otherwise False.
    """
    object_inventory_enum = target_state.conditional_placing.object_inventory_enum
    object_to_place = target_state.conditional_placing.object_to_place
    count_to_collect = target_state.conditional_placing.count_to_collect
    count_to_stand = target_state.conditional_placing.count_to_stand
    
    previous_state = gd.states[0]
    current_state = gd.states[1]
    
    # Check inventory in the previous state
    prev_inventory_check = check_inventory(previous_state.inventory, object_inventory_enum, count_to_collect)
    
    # Check inventory in the current state
    curr_inventory_check = check_inventory(current_state.inventory, object_inventory_enum, count_to_collect)
    
    # Check map (same as before)
    placed_check = check_map(current_state.map.game_map, object_to_place, count_to_stand)
    
    # Ensure that the sequence is correct:
    # 1) The required amount of items were NOT collected in the previous state
    # 2) The required amount of items WERE collected in the current state
    # 3) The required number of objects was placed on the map
    return jnp.logical_and(jnp.logical_and(jnp.logical_not(prev_inventory_check), curr_inventory_check), placed_check)
    