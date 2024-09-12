import random
from gym import Wrapper
import random
import importlib
import os
import jax
from craftext.checkers.base_functions.state_adapter import GameData

def load_scenarios():
    scenarios = {}
    scenarios_dir = "craftext/scenarios"
    for file in os.listdir(scenarios_dir):
            scenario_module_name = f"craftext.scenarios.{file}.instructions"
            scenario_module = importlib.import_module(scenario_module_name)
            scenarios.update(scenario_module.instructions)
    
    return scenarios

def load_test_scenarios(test_group='base'):
    scenarios = {}
    scenarios_dir = "craftext/scenarios"
    for file in os.listdir(scenarios_dir):
            if test_group in file:
                scenario_module_name = f"craftext.scenarios.{file}.instructions"
                scenario_module = importlib.import_module(scenario_module_name)
                scenarios.update(scenario_module.test)
    return scenarios
     
def get_random_instruction_and_checker():
    variable_name = "CRAFTEXT_RUN_MODE"
    value = os.getenv(variable_name)
    if value == "train":
        all_scenarios = load_scenarios()
    else: 
        variable_name = "TEST_GROUP"
        value = os.getenv(variable_name)
        all_scenarios = load_test_scenarios(value)

    print("Choose scnario:")
    for i in range(len(all_scenarios)):
         print(f"{i}) {list(all_scenarios.keys())[i]}")
         scenario_key =list(all_scenarios.keys())[i]
         scenario = all_scenarios[scenario_key]
         print("Instruction example: ", scenario['instruction'])

    idx = int(input("Input: "))
    random_scenario_key =list(all_scenarios.keys())[idx]
    random_scenario = all_scenarios[random_scenario_key]
    
    instructions = [random_scenario['instruction']] + random_scenario['instruction_paraphrases']
    random_instruction = instructions[0]
    keys = [0,0]
    return keys, random_instruction, random_scenario['check_lambda']


class InstructionWrapper(Wrapper):
    def __init__(self, env):
        super(InstructionWrapper, self).__init__(env)
        self.instruction_keys = None
        self.game_data = []
        self.actions = []

    def reset(self, _rng, env_params):
        # Выбираем случайную инструкцию
        self.actions = []
        instruction_keys, instruction_str, checker = get_random_instruction_and_checker()
        self.instruction_keys = instruction_keys
        self.instruction_str = instruction_str
        self.checker = checker

        # Сбрасываем основную среду
        obs, state = self.env.reset(_rng, env_params)
        self.game_data = [state]
        # Добавляем инструкцию к наблюдению
        obs_with_instruction = [obs,instruction_keys]
        return obs_with_instruction, state

    def step(self,  _rng, env_state, action, env_params):
        # Выполняем шаг основной среды
        obs, state, reward, done, info = self.env.step( _rng, env_state, action, env_params)

    #    # print("Right state: ", state.inventory)
    #     self.game_data = jax.numpy.array(self.game_data + [state])
    #     self.actions = jax.numpy.array(self.actions + [action])

        # Добавляем инструкцию к наблюдению
        obs_with_instruction = [obs,self.instruction_keys]

        done = self.checker(GameData(state, action))

        return obs_with_instruction, state, reward, done, info

if __name__=="__main__":
    instruction, checker = get_random_instruction_and_checker()
    print(instruction)