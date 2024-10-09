import os
import importlib
import yaml
import craftext

def get_configs_path():
    module_path = craftext.__spec__.submodule_search_locations[0]
    return os.path.join(module_path, 'configs')

def get_default_scenario_path():
    """Gets the default absolute path to the scenarios directory based on module installation."""
    module_path = craftext.__spec__.submodule_search_locations[0]
    return os.path.join(module_path, 'scenarios')

def parse_craftext_settings():
    """Parses the CRAFTEXT_SETTINGS environment variable and returns the mode, instruction type, and data key."""
    settings = os.getenv("CRAFTEXT_SETTINGS", "")
    if '&&' in settings:
        mode, instruction_type, data_key, test = settings.split('&&')
    else:
        mode = settings
        instruction_type = 'instruction'
        data_key = 'instructions'
        test = False
    return mode, instruction_type, data_key, bool(test)

def load_config_or_env(config_path=None):
    """Loads the configuration from a YAML file if provided, otherwise from environment variables."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            print("---- USE CONFIG CRAFTEXT ----")
        mode = config.get('dataset_key', 'default_mode')
        instruction_type = 'instruction' if not config.get('use_parafrases', False) else 'instruction_paraphrases'
        data_key = config.get('subset_key', 'instructions')
        enviroment_name = config.get('base_environment', 'Craftax-Pixels-v1-Text')
        test = config.get('test', False)
        
        if enviroment_name == 'Craftax-Pixels-v1-Text':
            environment_key = 1
            print("ENVIRONMENT KEY 1 - Craftax-Pixels-v1-Text")
        elif enviroment_name == 'Craftax-Classic-Pixels-v1-Text':
            print("ENVIRONMENT KEY 0 - Craftax-Classic-Pixels-v1-Text")
            environment_key = 0
    else:
        mode, instruction_type, data_key = parse_craftext_settings()
        environment_key = 0
    return mode, instruction_type, data_key, environment_key, test

def load_scenarios_by_mode(mode, data_key, test=False):
    """Loads scenarios based on the mode and the data key (e.g., instructions, small_train, other)."""
    scenarios = {}
    scenarios_dir = get_default_scenario_path()
    module = "test" if test else "instructions"
    
    if scenarios_dir is None:
        raise ValueError("Scenario path could not be determined. Please set the CRAFTEXT_SCENARIO_PATH environment variable.")

    for file in os.listdir(scenarios_dir):
        if mode in file:
            scenario_module_name = f"craftext.scenarios.{file}.{module}"
            scenario_module = importlib.import_module(scenario_module_name)
            if hasattr(scenario_module, data_key):
                scenarios.update(getattr(scenario_module, data_key))
    print(scenarios)
    return scenarios

def load_scenarios(config_name=None):
    config_path = get_configs_path()
    if config_name:
            config_name = str(os.path.join(config_path, config_name))+".yaml"
            print(config_name)
    else:
            config_name = None
    mode, instruction_type, data_key, environment_key, test = load_config_or_env(config_name)
    return load_scenarios_by_mode(mode, data_key, test), environment_key
