# craftext_scenarious.py

import os
import numpy as np
import jax
import jax.numpy as jnp
from craftext.craftext_encoder import EncodeModel, EncodeForm
from craftext.scenarios_loader import ScenariosConfig, ScenariosConfigLoader, load_scenarios #load_scenarios, parse_craftext_settings, load_config_or_env, get_configs_path

from dataclasses import dataclass
from jax import lax

@dataclass
class ScenarioData:
    instructions_list: list
    checkers_list: list
    indices_list: list
    encoded_instructions_list: list
    embeddings_list: list

@dataclass
class ScenarioDataJAX:
    encoded_instructions_list: jnp.array
    embeddings_list: jnp.array
    checkers_list: list

class CrafTextScenarios:
    def __init__(self, encode_model, config_name=None):
        """
        Initializes the CrafTextScenarios with an EncodeModel and scenario configuration.
        """
        self.encode_model = encode_model
        self.config = ScenariosConfigLoader().load_config(config_name)
       # self.config = self._load_config(config_name)
        self.use_parafrases =  self.config.use_parafrases
        self.environment_key = int("Classic" not in self.config.base_environment)
        self.all_scenario = self._load_scenarios(self.config)
        self.scenario_data = self._prepare_scenarios()
        self.scenario_data_jax = self.scenarios_to_jax()
    
    @property
    def initial_instruction(self):
        """
        Generates the default encoded instruction for initializing network parameters.
        """
        return self.encode_model.encode("None")

    # def _load_config(self, config_name):
    #     """
    #     Loads configuration settings for scenarios, based on a YAML config or an environment variable.
    #     """
    #     if config_name is None:
    #         return None
    #     config_path = get_configs_path()
    #     config_name = str(os.path.join(config_path, config_name)) + ".yaml"
    #     config = load_config_or_env(config_name)
    #     return config
    
    def _load_scenarios(self, config):
        """
        Loads scenarios from a specified configuration file.
        """
        return load_scenarios(config)

    def get_scenarios(self):
        """
        Retrieves the processed scenario data.
        """
        return self.scenario_data

    def _prepare_scenarios(self):
        """
        Prepares and encodes the scenarios, deciding whether to use embeddings or tokens.
        """
        instructions_list, checkers_list, indices_list = [], [], []
        encoded_instructions_list, embeddings_list = [], []

        for idx, (key, scenario) in enumerate(self.all_scenario.items()):
            instructions = [scenario.get('instruction')]
            checkers = [scenario.get('check_lambda')]

            if self.use_parafrases and 'instruction_paraphrases' in scenario:
                instructions += scenario['instruction_paraphrases']
                checkers += [scenario['check_lambda']] * len(scenario['instruction_paraphrases'])

            for instruction, checker in zip(instructions, checkers):
                encoded_instruction = self.encode_model.encode(instruction)

                instructions_list.append(instruction)
                checkers_list.append(checker)
                indices_list.append(idx)
                if self.encode_model.form_to_use == EncodeForm.EMBEDDING:
                    embeddings_list.append(encoded_instruction)
                else:
                    encoded_instructions_list.append(encoded_instruction)

        return ScenarioData(
            instructions_list=instructions_list,
            checkers_list=checkers_list,
            indices_list=np.array(indices_list).reshape(len(instructions_list), 1),
            encoded_instructions_list=np.array(encoded_instructions_list).reshape(len(instructions_list), -1) if encoded_instructions_list else None,
            embeddings_list=np.array(embeddings_list).reshape(len(instructions_list), -1) if embeddings_list else None
        )

    def scenarios_to_jax(self):
        """
        Converts scenario data to JAX-compatible structures.
        """
        encoded_instructions_jax = jnp.array(self.scenario_data.encoded_instructions_list) if self.scenario_data.encoded_instructions_list is not None else None
        embeddings_jax = jnp.array(self.scenario_data.embeddings_list) if self.scenario_data.embeddings_list is not None else None
        checkers_list_jax = self._prepare_jax_checkers(self.scenario_data.checkers_list)
        return ScenarioDataJAX(
            encoded_instructions_list=encoded_instructions_jax,
            embeddings_list=embeddings_jax,
            checkers_list=checkers_list_jax
        )

    def _prepare_jax_checkers(self, checkers_list):
        """
        Converts checkers list to JAX-compatible functions using `lax.switch` and `vmap`.
        """
        indices = jnp.arange(len(checkers_list))
        def apply_checker(i, x, y):
            return lax.switch(i, checkers_list, x, y)
        vmap_checkers = jax.vmap(apply_checker, in_axes=(None, None, None))
        return vmap_checkers
