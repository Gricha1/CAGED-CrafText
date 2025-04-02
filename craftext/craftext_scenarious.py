# craftext_scenarious.py

import json

import time
import numpy as np
import jax
import jax.numpy as jnp
from craftext.encoders.craftext_base_model_encoder import EncodeModel, EncodeForm
from craftext.scenarios_loader import ScenariosConfig, ScenariosConfigLoader, load_scenarios #load_scenarios, parse_craftext_settings, load_config_or_env, get_configs_path
from craftext.scenarios.constants import plans_path
from dataclasses import dataclass
from jax import lax
from tqdm import tqdm


@dataclass
class ScenarioData:
    instructions_list: list
    checkers_list: list
    str_check_lambda_list: list
    indices_list: list
    scenario_names: list
    embeddings_list: list

@dataclass
class ScenarioDataJAX:
    embeddings_list: jax.Array
    checkers_list: list

class CrafTextScenarios:
    def __init__(self, encode_model, config_name=None, use_plans=False):
        """
        Initializes the CrafTextScenarios with an EncodeModel and scenario configuration.
        """
        self.encode_model = encode_model
        self.config = ScenariosConfigLoader().load_config(config_name)
        self.use_parafrases =  self.config.use_parafrases
        self.environment_key = 0 if "Classic" in self.config.base_environment else 1 # int("Classic" not in self.config.base_environment)
       # print(self.config.base_environment)
        #exit()
        self.n_instructions = 0
        self.use_plans = use_plans
        self.instruction_to_apdate_file = plans_path
        self.all_scenario = self._load_scenarios(self.config)
        self.scenario_data = self._prepare_scenarios()
        self.scenario_data_jax = self.scenarios_to_jax()
    
    @property
    def initial_instruction(self):
        """
        Generates the default encoded instruction for initializing network parameters.
        """
        return self.encode_model.encode(["None"])[:1]

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

    def encode(self, instruction):
        return [self.encode_model.encode(instruction)]

    def _load_original_scenarios(self, instruction_to_apdate_file=None):
        instructions_list, checkers_list, indices_list, scenario_names_list = [], [], [], []

        for idx, (key, scenario) in tqdm(enumerate(self.all_scenario.items())):
            # Collect instructions and their corresponding checkers
            print(scenario)
            instructions = [scenario.get('instruction')]
            checkers = [scenario.get('check_lambda')]
            names = [scenario.get('str_check_lambda')]

            if self.use_parafrases and 'instruction_paraphrases' in scenario:
                instructions += scenario['instruction_paraphrases']
                checkers += [scenario['check_lambda']] * len(scenario['instruction_paraphrases'])
                names += [scenario['str_check_lambda']] * len(scenario['instruction_paraphrases'])
            for instruction, checker in zip(instructions, checkers):
                instructions_list.append(instruction)
                checkers_list.append(checker)
                indices_list.append(idx)
                scenario_names_list.append(names)

        # OPTIONALLY: Update instructions to predefined plans

        # with open("instructions_new_obj.json", 'w', encoding='utf-8') as f:
        # #         json.dump(instructions_list, f, ensure_ascii=False, indent=4)

        #easy_gpt4_action_plans
        if self.use_plans:
            with open(self.instruction_to_apdate_file, 'r', encoding='utf-8') as f:
                action_plans = json.load(f)
            instructions_list = [action_plans[instr] if instr in action_plans else "none" for instr in instructions_list ]
            print("="*40)
            print()
            print()
            print("Use preloaded plans in craftext_scenarios.py")
            print()
            print("="*40)
            print("Encode instructions!")
            time.sleep(14)
        return instructions_list, checkers_list, indices_list, scenario_names_list

    def _prepare_scenarios(self):
        """
        Prepares and encodes the scenarios, deciding whether to use embeddings or tokens,
        while handling multiple embeddings per instruction.
        """
        instructions_list, checkers_list, indices_list, scenario_names_list = self._load_original_scenarios()
        # Process instructions in batches
        
        instructions_list_f, checkers_list_f, indices_list_f, embeddings_list_f, scenario_names_f = [], [], [], [], []
        batch_size = 5
        for i in tqdm(range(0, len(instructions_list), batch_size)):
            batch_instructions = instructions_list[i:i+batch_size]
            batch_checkers = checkers_list[i:i+batch_size]
            batch_indices = indices_list[i:i+batch_size]
            batch_names = scenario_names_list[i:i+batch_size]

            # Encode instructions in the batch
            encoded_instructions = self.encode_model.encode(batch_instructions)

            # Determine the number of variants per instruction
            num_variants = len(encoded_instructions) // len(batch_instructions)
            assert len(encoded_instructions) == len(batch_instructions) * num_variants, \
                f"Unexpected size of encoded instructions(instructions len - {len(encoded_instructions)} and batch len - {len(batch_instructions)}). Ensure encode_model is consistent."

            # Process each instruction's embeddings and replicate corresponding metadata
            for j, instruction in enumerate(batch_instructions):
                for k in range(num_variants):
                    variant_index = j * num_variants + k
                    instructions_list_f.append(instruction)
                    checkers_list_f.append(batch_checkers[j])
                    indices_list_f.append(batch_indices[j])
                    embeddings_list_f.append(encoded_instructions[variant_index])
                    scenario_names_f.append(batch_names[j])

        scenario_data = ScenarioData(
            instructions_list=instructions_list_f,
            checkers_list=checkers_list_f,
            str_check_lambda_list = scenario_names_f,
            scenario_names = scenario_names_f,
            indices_list=np.array(indices_list_f).reshape(-1, 1),
            embeddings_list=np.array(embeddings_list_f).reshape(len(embeddings_list_f), -1) if embeddings_list_f else None
        )
        # Return the prepared ScenarioData
        return scenario_data
        


    def scenarios_to_jax(self):
        """
        Converts scenario data to JAX-compatible structures.
        """
       # encoded_instructions_jax = jnp.array(self.scenario_data.encoded_instructions_list) if self.scenario_data.encoded_instructions_list is not None else None
        embeddings_jax = jnp.array(self.scenario_data.embeddings_list) if self.scenario_data.embeddings_list is not None else None
        checkers_list_jax = self._prepare_jax_checkers(self.scenario_data.checkers_list)
        return ScenarioDataJAX(
          #  encoded_instructions_list=encoded_instructions_jax,
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

def create_scenarios_with_dataset(use_plans_gpt):
    class CustomCrafTextScenariosWithPlans(CrafTextScenarios):
        def __init__(self, encode_model, config_name):
            super().__init__(encode_model, config_name=config_name, use_plans=use_plans_gpt)
    return CustomCrafTextScenariosWithPlans