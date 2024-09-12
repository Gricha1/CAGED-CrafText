from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
import numpy as np
import jax.numpy as jnp

@dataclass
class PlayerVariables:
    player_position: Optional[jnp.ndarray] = None
    player_level: Optional[int] = None
    player_direction: Optional[int] = None
    player_health: Optional[float] = None
    player_food: Optional[int] = None
    player_drink: Optional[int] = None
    player_energy: Optional[int] = None
    player_mana: Optional[int] = None
    is_sleeping: Optional[bool] = None
    is_resting: Optional[bool] = None
    player_recover: Optional[float] = None
    player_hunger: Optional[float] = None
    player_thirst: Optional[float] = None
    player_fatigue: Optional[float] = None
    player_recover_mana: Optional[float] = None
    player_xp: Optional[int] = None
    player_dexterity: Optional[int] = None
    player_strength: Optional[int] = None
    player_intelligence: Optional[int] = None
    learned_spells: Optional[jnp.ndarray] = None
    sword_enchantment: Optional[int] = None
    bow_enchantment: Optional[int] = None
    boss_progress: Optional[int] = None
    boss_timesteps_to_spawn_this_round: Optional[int] = None
    light_level: Optional[float] = None
    state_rng: Optional[jnp.ndarray] = None
    timestep: Optional[int] = None

@dataclass
class PlayerAchievements:
    achievements: Optional[List[str]] = None

@dataclass
class PlayerInventory:
    wood: Optional[jnp.ndarray] = None
    stone: Optional[jnp.ndarray] = None
    coal: Optional[jnp.ndarray] = None
    iron: Optional[jnp.ndarray] = None
    pickaxe: Optional[jnp.ndarray] = None
    sword: Optional[jnp.ndarray] = None
    armour: Optional[jnp.ndarray] = None
    potions: Optional[jnp.ndarray] = None

@dataclass
class GameMap:
    game_map: Optional[jnp.ndarray] = None

    def look_around(self, position):
        if self.game_map is None:
            return None
        r = 4
        x, y = position
        map_around = self.game_map[x-r:x+r+1, y-r:y+r+1]
        unique_objects = jnp.unique(map_around)
        unique_object_indices = [blocks_list.index(item) for item in unique_objects]
        return unique_object_indices

@dataclass
class PlayerState:
    variables: PlayerVariables
    achievements: Optional[PlayerAchievements] = None
    inventory: Optional[PlayerInventory] = None
    map: Optional[GameMap] = None
    action: Optional[int] = None

    @classmethod
    def from_state(cls, state, action):
        action = action
        variables = PlayerVariables(
            player_position=jnp.array(state.player_position) if hasattr(state, 'player_position') else None,
            player_level=state.player_level,
            player_direction=state.player_direction,
            player_health=state.player_health,
            player_food= state.player_food,
            player_drink=state.player_drink,
            player_energy=state.player_energy,
            player_mana=state.player_mana,
            is_sleeping=state.is_sleeping,
            is_resting=state.is_resting,
            player_recover=state.player_recover,
            player_hunger=state.player_hunger,
            player_thirst=state.player_thirst,
            player_fatigue=state.player_fatigue,
            player_recover_mana=state.player_recover_mana,
            player_xp=state.player_xp,
            player_dexterity=state.player_dexterity,
            player_strength=state.player_strength,
            player_intelligence=state.player_intelligence,
            learned_spells=state.learned_spells,
            sword_enchantment=state.sword_enchantment,
            bow_enchantment=state.bow_enchantment,
            boss_progress=state.boss_progress,
            light_level=state.light_level,
            state_rng=state.state_rng,
            timestep=state.timestep,
        )

        achievements = PlayerAchievements(
            achievements=state.achievements
        )

        inventory = PlayerInventory(
            wood=jnp.array(state.inventory.wood),
            stone=jnp.array(state.inventory.stone),
            coal=jnp.array(state.inventory.coal),
            iron=jnp.array(state.inventory.iron),
            pickaxe=jnp.array(state.inventory.pickaxe),
            sword=jnp.array(state.inventory.sword),
            armour=jnp.array(state.inventory.armour),
            potions=jnp.array(state.inventory.potions),
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


class GameData:
    def __init__(self, state, action):
        self.states = [PlayerState.from_state(state, action)]
        
    