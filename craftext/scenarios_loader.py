import os
import importlib
import random
import importlib.util
import craftext

def get_default_scenario_path():
    """Gets the default absolute path to the scenarios directory based on module installation."""
    module_path = craftext.__spec__.submodule_search_locations[0]
    return os.path.join(module_path, 'scenarios')


def parse_craftext_settings():
    """Parses the CRAFTEXT_SETTINGS environment variable and returns the mode, instruction type, and data key."""
    settings = os.getenv("CRAFTEXT_SETTINGS", "")
    if '&&' in settings:
        mode, instruction_type, data_key = settings.split('&&')
    else:
        mode = settings
        instruction_type = 'instruction'
        data_key = 'instructions'
    return mode, instruction_type, data_key

def load_scenarios():
    mode, instruction_type, data_key = parse_craftext_settings()
    return load_scenarios_by_mode(mode, data_key)
    
def load_scenarios_by_mode(mode, data_key):
    """Loads scenarios based on the mode and the data key (e.g., instructions, small_train, other)."""
    scenarios = {}
    scenarios_dir = get_default_scenario_path() #os.getenv("CRAFTEXT_SCENARIO_PATH", get_default_scenario_path())
    
    if scenarios_dir is None:
        raise ValueError("Scenario path could not be determined. Please set the CRAFTEXT_SCENARIO_PATH environment variable.")

    for file in os.listdir(scenarios_dir):
        if mode in file:
            scenario_module_name = f"craftext.scenarios.{file}.instructions"
            scenario_module = importlib.import_module(scenario_module_name)
            if hasattr(scenario_module, data_key):
                scenarios.update(getattr(scenario_module, data_key))
    
    return scenarios

def sample_instruction(scenarios, instruction_type):
    """Samples an instruction depending on the instruction type."""
    random_scenario_key = random.choice(list(scenarios.keys()))
    random_scenario = scenarios[random_scenario_key]
    
    if instruction_type == 'pure_instruction':
        instruction = random_scenario['instruction']
    else:
        instructions = [random_scenario['instruction']] + random_scenario.get('instruction_paraphrases', [])
        instruction = random.choice(instructions)
    
    return instruction, random_scenario['check_lambda']

def get_random_instruction_and_checker():
    """Fetches a random instruction and its corresponding check function."""
    mode, instruction_type, data_key = parse_craftext_settings()
    all_scenarios = load_scenarios_by_mode(mode, data_key)
    random_instruction, check_lambda = sample_instruction(all_scenarios, instruction_type)
    
    return random_instruction, check_lambda
