from flax.struct import dataclass, field
from craftext.environment.craftext_constants import MobType
from craftext.environment.scenarious.checkers.target_state import TargetState
#from craftext.environment.scenarious.checkers.target_state_cmdp_relactional_point_of_intereset import TargetOfInterest, TargetOfInterest
#from craftext.environment.scenarious.checkers.target_state_cmdp_relactional_avoid_mob import AvoidMobDistance
from jax import numpy as jnp
import jax

@dataclass
class AvoidMobDistance:
    mob: MobType = field(default_factory=MobType.ZOMBIE)
    distance: int = field(default_factory=int)
    
@dataclass
class TypeOfInterest:
    FOOD = 0,
    WATER = 1,
    TREE = 2,
    STONE = 3
    
@dataclass
class TargetOfInterest:
    object_of_interest: int = field(default_factory=TypeOfInterest.FOOD)
    far_from_agent: int = field(default_factory=5)
    last_visible_target_position: jax.Array = field(default_factory=jnp.zeros(shape=(1, 2), dtype=jnp.int16))
    
@dataclass
class CMDPTargetState(TargetState):
    avoid_mob_distance: AvoidMobDistance = field(default_factory=AvoidMobDistance)
    target_of_interest_water: TargetOfInterest = field(default_factory=TargetOfInterest)
    target_of_interest_food: TargetOfInterest = field(default_factory=TargetOfInterest)
    
    
    
