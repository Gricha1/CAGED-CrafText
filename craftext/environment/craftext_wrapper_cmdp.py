from typing import Any, Optional
import jax
import jax.numpy as jnp
from jax import lax

from flax import struct
from gym import Wrapper


from craftext.environment.encoders.craftext_base_model_encoder import EncodeForm
from craftext.environment.encoders.craftext_distilbert_model_encoder import DistilBertEncode

from craftext.environment.scenarious.manager import ScenariosNoLambda

from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic
from craftext.environment.craftext_constants import Scenarios 
from craftext.environment.scenarious.checkers.achivments       import checker_acvievments
from craftext.environment.scenarious.checkers.time_constrained import checker_time_placement
from craftext.environment.scenarious.checkers.building_star    import checker_star
from craftext.environment.scenarious.checkers.building_line    import checker_line
from craftext.environment.scenarious.checkers.building_square  import checker_square
from craftext.environment.scenarious.checkers.conditional      import checker_conditional_placement
from craftext.environment.scenarious.checkers.relevant         import cheker_localization
from craftext.environment.scenarious.checkers.target_state     import TargetState
from typing import Union

from craftext.environment.craftext_wrapper import InstructionWrapper


class CMDPInstructionWrapper(InstructionWrapper):
    def step(self, _rng, env_state, action, env_params):
        obs, state, reward, done, info = super().step(_rng, env_state, action, env_params)
        info["cost"] = 1
        return obs, state, reward, done, info