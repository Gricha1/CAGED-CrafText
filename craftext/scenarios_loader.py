import os
import importlib
import yaml
import craftext


import yaml
import os
import craftext  # Предполагая, что модуль craftext уже импортирован

class ScenariosConfig:
    def __init__(self, dataset_key, subset_key, base_environment, use_parafrases, test):
        self.dataset_key = dataset_key # the task type 
        self.subset_key = subset_key # the task complexity and if test parafrases / items
        self.base_environment = base_environment # use Classic or not
        self.use_parafrases = use_parafrases # use parafrases in loading or not
        self.test = test # is it test data or not

    def __repr__(self):
        return (f"ScenariosConfig(dataset_key={self.dataset_key}, subset_key={self.subset_key}, "
                f"base_environment={self.base_environment}, use_parafrases={self.use_parafrases}, test={self.test})")


class ScenariosConfigLoader:
    @staticmethod
    def get_config_path(config_name: str) -> str:
        module_path = craftext.__spec__.submodule_search_locations[0]
        config_path = os.path.join(module_path, 'configs', f"{config_name}.yaml")
        return config_path

    @staticmethod
    def load_config(config_name: str) -> ScenariosConfig:
        config_path = ScenariosConfigLoader.get_config_path(config_name)
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        return ScenariosConfig(
            dataset_key=config_data.get("dataset_key"),
            subset_key=config_data.get("subset_key"),
            base_environment=config_data.get("base_environment"),
            use_parafrases=config_data.get("use_parafrases", False),
            test=config_data.get("test", False)
        )


def get_default_scenario_path():
    """Gets the default absolute path to the scenarios directory based on module installation."""
    module_path = craftext.__spec__.submodule_search_locations[0]
    return os.path.join(module_path, 'scenarios')

def load_scenarios(scenarious_config):
    scenarios = {}
    scenarios_dir = get_default_scenario_path()
    module = "test" if scenarious_config.test else "instructions"
    mode = scenarious_config.dataset_key
    data_key = scenarious_config.subset_key
    if scenarios_dir is None:
        raise ValueError("Scenario path could not be determined.")

    for file in os.listdir(scenarios_dir):
        if mode in file:
            scenario_module_name = f"craftext.scenarios.{file}.{module}"
            scenario_module = importlib.import_module(scenario_module_name)
            if hasattr(scenario_module, data_key):
                scenarios.update(getattr(scenario_module, data_key))
    print(scenarios)
    return scenarios


    



# def get_configs_path():
#     module_path = craftext.__spec__.submodule_search_locations[0]
#     return os.path.join(module_path, 'configs')

# def get_default_scenario_path():
#     """Gets the default absolute path to the scenarios directory based on module installation."""
#     module_path = craftext.__spec__.submodule_search_locations[0]
#     return os.path.join(module_path, 'scenarios')

# def parse_craftext_settings():
#     """Parses the CRAFTEXT_SETTINGS environment variable and returns the mode, instruction type, and data key."""
#     settings = os.getenv("CRAFTEXT_SETTINGS", "")
#     if '&&' in settings:
#         mode, instruction_type, data_key, test = settings.split('&&')
#     else:
#         mode = settings
#         instruction_type = 'instruction'
#         data_key = 'instructions'
#         test = False
#     return mode, instruction_type, data_key, bool(test)

# def load_config_or_env(config_path=None):
#     """Loads the configuration from a YAML file if provided, otherwise from environment variables."""
#     if config_path and os.path.exists(config_path):
#         with open(config_path, 'r') as file:
#             config = yaml.safe_load(file)
#             print("---- USE CONFIG CRAFTEXT ----")
#         mode = config.get('dataset_key', 'default_mode')
#         instruction_type = 'instruction' if not config.get('use_parafrases', False) else 'instruction_paraphrases'
#         data_key = config.get('subset_key', 'instructions')
#         enviroment_name = config.get('base_environment', 'Craftax-Pixels-v1-Text')
#         test = config.get('test', False)
        
#         if enviroment_name == 'Craftax-Pixels-v1-Text':
#             environment_key = 1
#             print("ENVIRONMENT KEY 1 - Craftax-Pixels-v1-Text")
#         elif enviroment_name == 'Craftax-Classic-Pixels-v1-Text':
#             print("ENVIRONMENT KEY 0 - Craftax-Classic-Pixels-v1-Text")
#             environment_key = 0
#     else:
#         mode, instruction_type, data_key = parse_craftext_settings()
#         environment_key = 0
#     return mode, instruction_type, data_key, environment_key, test

# def load_scenarios_by_mode(mode, data_key, test=False):
#     """Loads scenarios based on the mode and the data key (e.g., instructions, small_train, other)."""
#     scenarios = {}
#     scenarios_dir = get_default_scenario_path()
#     module = "test" if test else "instructions"
    
#     if scenarios_dir is None:
#         raise ValueError("Scenario path could not be determined. Please set the CRAFTEXT_SCENARIO_PATH environment variable.")

#     for file in os.listdir(scenarios_dir):
#         if mode in file:
#             scenario_module_name = f"craftext.scenarios.{file}.{module}"
#             scenario_module = importlib.import_module(scenario_module_name)
#             if hasattr(scenario_module, data_key):
#                 scenarios.update(getattr(scenario_module, data_key))
#     print(scenarios)
#     return scenarios

# def load_scenarios(config_name=None):
#     config_path = get_configs_path()
#     if config_name:
#             config_name = str(os.path.join(config_path, config_name))+".yaml"
#             print(config_name)
#     else:
#             config_name = None
#     mode, instruction_type, data_key, environment_key, test = load_config_or_env(config_name)
#     return load_scenarios_by_mode(mode, data_key, test), environment_key
