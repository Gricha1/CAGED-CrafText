import jax

from jax import (
    numpy as jnp,
    lax
)

from typing import Union
from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic

from craftext.environment.scenarious.checkers.target_state_cmdp_build_budget_collect import BuildBudgetState
from craftext.environment.craftext_constants import BlockType

def checker_budget_build_collect(game_data: Union[GameDataClassic, GameData],  target_state: BuildBudgetState) -> jax.Array:
    # raise NotImplementedError("checker_budget_build_collect is not implemented yet")
    block_type = target_state.block_type
    return new_item(game_data, block_type)

def new_item(game_data: Union[GameDataClassic, GameData], block_type: BlockType):
    block_type  = block_type

    def get_item(index, inventory):
        return lax.switch(index,
            [
                lambda: inventory.wood,
                lambda: inventory.stone,
                lambda: inventory.coal,
                lambda: inventory.iron,
                lambda: inventory.diamond,
                lambda: inventory.sapling,
                lambda: inventory.wood_pickaxe,
                lambda: inventory.stone_pickaxe,
                lambda: inventory.iron_pickaxe,
                lambda: inventory.wood_sword,
                lambda: inventory.stone_sword,
                lambda: inventory.iron_sword,
                
                # lambda: lax.switch(inventory.inventory, [lambda: inventory.pickaxe, lambda: inventory.wood_pickaxe]), 
                # lambda: lax.switch(inventory.inventory, [lambda: inventory.sword, lambda: inventory.stone_pickaxe]),
                # lambda: lax.switch(inventory.inventory, [lambda: inventory.bow,   lambda: inventory.iron_pickaxe]), 
                # lambda: lax.switch(inventory.inventory, [lambda: inventory.arrows, lambda: inventory.wood_sword]),
                # lambda: lax.switch(inventory.inventory, [lambda: inventory.armour, lambda: inventory.stone_sword]),
                # lambda: lax.switch(inventory.inventory, [lambda: inventory.torches, lambda: inventory.iron_sword]),
            
                # lambda: inventory.ruby,
                # lambda: inventory.sapphire,
                
                # lambda: inventory.diamond, #lambda: inventory.potions
                # lambda: inventory.diamond, #lambda: inventory.books
            ]
        )
    
    
    
    
    curr = get_item(block_type, game_data.states[0].inventory)
    prev = get_item(block_type, game_data.states[1].inventory)
    return curr - prev > 0


