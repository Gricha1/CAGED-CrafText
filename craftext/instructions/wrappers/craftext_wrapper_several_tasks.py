# instruction_wrapper.py

from typing import Any, Optional
import jax
import jax.numpy as jnp
from flax import linen as struct
from gym import Wrapper

from craftext.encoders.craftext_base_model_encoder import EncodeForm
from craftext.encoders.craftext_distilbert_model_encoder import DistilBertEncode

from craftext.instructions.scenarios.handlers.craftext_scenarious_no_lambda import ScenariosNoLambda
from craftext.adapters.state_adapter import GameData

from craftext.adapters.state_adapter_classic import GameDataClassic

# from craftext.checkers_jax.time_constrained import at_time_block_placed
# from craftext.checkers_jax.building_star import is_cross_formed
from jax import tree_util

@struct.dataclass
class TextEnvState:
    env_state: Any
    timestep: int
    instruction: Optional[jax.Array]
    idx: int
    success_rate: float
    total_success_rate: float
    environment_key: int
    rng: int
    
def get_checker_functions():
    from craftext.checkers_jax.time_constrained import at_time_block_placed
    from craftext.checkers_jax.building_star import is_cross_formed
    from craftext.checkers_jax.building_line import is_line_formed
    from craftext.checkers_jax.building_line import is_square_formed
    from craftext.checkers_jax.conditional import conditional_placing
    from craftext.checkers_jax.relevant import place_object_relevant_to
    
    return [
        conditional_placing,
        place_object_relevant_to,
        is_line_formed,
        is_square_formed,
        is_cross_formed,
        at_time_block_placed
    ]
class InstructionWrapperSeveralTasks(Wrapper):
    def __init__(self, env , config_name=None, scenario_handler_class=ScenariosNoLambda,
                  encode_model_class=DistilBertEncode, encode_form=EncodeForm.EMBEDDING):
        """
        Initializes the InstructionWrapper with the environment, creating EncodeModel and CrafTextScenarios.
        
        Parameters:
        - env: The environment to wrap.
        - config_name: Optional configuration name for scenarios.
        - encode_model_class: A class for the encoding model. Defaults to DistilBertEncode.
        - encode_form: The form of encoding (EMBEDDING or TOKEN). Defaults to EMBEDDING.
        """
        super().__init__(env)

        # Initialize the encoding model using the provided class
        self.encode_model = encode_model_class(form_to_use=encode_form)

        # Initialize the scenario handler with the encoding model
        self.scenario_handler = scenario_handler_class(self.encode_model, config_name)
        self.encoded_instruction = self.scenario_handler.initial_instruction
        self.scenario_arguments = (self.scenario_handler.scenario_data_jax.arguments)
        self.batched_scenario_args = tree_util.tree_map(
            lambda *xs: jnp.stack(xs),
            *self.scenario_arguments
        )
        self.env = env
        self.steps = 0

        # Determine the environment key and state structure
        self.environment_key = self.scenario_handler.environment_key
        self.StateStructure = GameData if self.environment_key == 1 else GameDataClassic

        print("Initialized Instruction Wrapper with environment key:", self.environment_key)
        print(self.StateStructure)
        self.n_instructions = len(self.scenario_handler.scenario_data.instructions_list)
        print(self.scenario_handler.scenario_data.instructions_list)
        print(len(self.scenario_handler.scenario_data.instructions_list))
        
        
        self.checkers = list(map(lambda x:  jax.vmap(x, in_axes=(None, 0)), get_checker_functions()))
        
        #print(self.scenario_handler.scenario_data_jax.arguments)
        #exit()
    
    def reset(self, _rng, env_params, instruction_idx=-1):
        """
        Resets the environment and selects a random instruction embedding or token for the new episode.
        """

        obs, state = self.env.reset(_rng, env_params)
        
        idx = jax.lax.cond(
                instruction_idx == -1, 
                lambda: jax.random.randint(_rng, shape=(), minval=0, maxval=len(self.scenario_handler.scenario_data_jax.embeddings_list)),
                lambda: instruction_idx
            )
        instructions_emb = self.scenario_handler.scenario_data_jax.embeddings_list[idx]

        # Initialize the state with the selected instruction embedding/token and set success rates to zero
        state = TextEnvState(
            env_state=state,
            timestep=state.timestep,
            instruction=instructions_emb,
            idx=idx,
            environment_key=self.environment_key,
            success_rate=0.0,
            total_success_rate=0.0,
            rng=_rng
        )
        return obs, state

    def step(self, _rng, env_state, action, env_params):
        """
        Takes a step in the environment, checking if the instruction is done, updating success rate and rewards.
        """
        obs, state, reward, done, info = self.env.step(_rng, env_state.env_state, action, env_params)
        # Obtain the game data vector for the current state and check instruction completion
        game_data_vector = self.StateStructure.from_state(env_state.env_state, state, action)

        results = jnp.array(
            [func(game_data_vector, self.batched_scenario_args) for func in self.checkers]
        )
        print(results)
        print(env_state.idx)
        print(done)
        instructions_done = jax.lax.dynamic_slice(results, (0, env_state.idx), (results.shape[0], 1))
        print(instructions_done)
        any_instruction_done = jnp.any(instructions_done)
        
        reward /= 50
        reward = jax.lax.cond(any_instruction_done, lambda _: reward + 1, lambda _: reward, operand=None)
        done = any_instruction_done | done
   
        new_episode_sr = env_state.success_rate + jnp.float32(any_instruction_done)

        # Update state with the new success rates
        state = TextEnvState(
            env_state=state,
            timestep=state.timestep,
            instruction=env_state.instruction,
            idx=env_state.idx,
            environment_key=env_state.environment_key,
            success_rate=new_episode_sr * (1 - done),
            total_success_rate=env_state.total_success_rate * (1 - done) + new_episode_sr * done,
            rng=env_state.rng
        )
        
        # Update step information in info dictionary
        info.update({"SR": state.total_success_rate, "steps": self.steps})
        self.steps += 1
        return obs, state, reward, done, info
 
#TODO:
# враппер для проверки всех задачь разом.
# посмотреть trl GRPO trainer -> использовать transformers
# ONE -> вернуть.