from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
import numpy as np
import jax.numpy as jnp
from flax import struct

@struct.dataclass
class PlayerVariables:
    player_position: Optional[jnp.ndarray] = None
    player_direction: Optional[int] = None
    player_health: Optional[int] = None
    player_food: Optional[int] = None
    player_drink: Optional[int] = None
    player_energy: Optional[int] = None
    is_sleeping: Optional[bool] = None
    player_recover: Optional[float] = None
    player_hunger: Optional[float] = None
    player_thirst: Optional[float] = None
    player_fatigue: Optional[float] = None
    light_level: Optional[float] = None
    state_rng: Optional[jnp.ndarray] = None
    timestep: Optional[int] = None

@struct.dataclass
class PlayerAchievements:
    achievements: Optional[List[str]] = None

@struct.dataclass
class PlayerInventory:
    inventory = 1
    wood: Optional[int] = None
    stone: Optional[int] = None
    coal: Optional[int] = None
    iron: Optional[int] = None
    diamond: Optional[int] = None
    sapling: Optional[int] = None
    wood_pickaxe: Optional[int] = None
    stone_pickaxe: Optional[int] = None
    iron_pickaxe: Optional[int] = None
    wood_sword: Optional[int] = None
    stone_sword: Optional[int] = None
    iron_sword: Optional[int] = None
     # Just for jax for correct invemtory check
    pickaxe: Optional[jnp.ndarray] = None
    sword: Optional[jnp.ndarray] = None
    bow: Optional[jnp.ndarray] = None
    arrows: Optional[jnp.ndarray] = None
    armour: Optional[jnp.ndarray] = None
    torches: Optional[jnp.ndarray] = None
    ruby: Optional[jnp.ndarray] = None
    sapphire: Optional[jnp.ndarray] = None
    potions: Optional[jnp.ndarray] = None
    books: Optional[jnp.ndarray] = None



@struct.dataclass
class GameMap:
    game_map: Optional[jnp.ndarray] = None


@struct.dataclass
class PlayerState:
    variables: PlayerVariables
    achievements: Optional[PlayerAchievements] = None
    inventory: Optional[PlayerInventory] = None
    map: Optional[GameMap] = None
    action: Optional[int] = None

    @classmethod
    def from_state(cls, state, action):
        variables = PlayerVariables(
            player_position=jnp.array(state.player_position) if hasattr(state, 'player_position') else None,
            player_direction=state.player_direction,
            player_health=state.player_health,
            player_food=state.player_food,
            player_drink=state.player_drink,
            player_energy=state.player_energy,
            is_sleeping=state.is_sleeping,
            player_recover=state.player_recover,
            player_hunger=state.player_hunger,
            player_thirst=state.player_thirst,
            player_fatigue=state.player_fatigue,
            light_level=state.light_level,
            state_rng=state.state_rng,
            timestep=state.timestep,
        )

        achievements = PlayerAchievements(
            achievements=jnp.array(state.achievements) if hasattr(state, 'achievements') else None
        )

        inventory = PlayerInventory(
            wood=state.inventory.wood,
            stone=state.inventory.stone,
            coal=state.inventory.coal,
            iron=state.inventory.iron,
            diamond=state.inventory.diamond,
            sapling=state.inventory.sapling,
            wood_pickaxe=state.inventory.wood_pickaxe,
            stone_pickaxe=state.inventory.stone_pickaxe,
            iron_pickaxe=state.inventory.iron_pickaxe,
            wood_sword=state.inventory.wood_sword,
            stone_sword=state.inventory.stone_sword,
            iron_sword=state.inventory.iron_sword,
            # Just for jax for correct invemtory check
            pickaxe=state.inventory.iron,
            sword=state.inventory.iron,
            bow=state.inventory.iron,
            arrows=state.inventory.iron,
            armour=state.inventory.iron,
            torches=state.inventory.iron,
            ruby=state.inventory.iron,
            sapphire=state.inventory.iron,
            potions=state.inventory.iron,
            books=state.inventory.iron,
        )

        game_map = GameMap(
            game_map=jnp.array(state.map) if hasattr(state, 'map') else None
        )

        return cls(
            variables=variables,
            achievements=achievements,
            inventory=inventory,
            map=game_map,
            action=action
        )

@struct.dataclass
class GameDataClassic:
    states: list

    @classmethod
    def from_state(cls, previos_state, current_state, action):
        player_state_current = PlayerState.from_state(current_state, action)
        player_state_previos = PlayerState.from_state(previos_state, action)
        return cls(states=[player_state_current, player_state_previos])
    