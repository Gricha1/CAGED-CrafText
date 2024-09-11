import random
from gym import Wrapper
import random
import importlib
import os
from craftext.checkers.base_functions.state_adapter import GameData

def load_scenarios():
    scenarios = {}
    scenarios_dir = "craftext/scenarios"
    
    # Загрузка всех сценариев из папки
    for file in os.listdir(scenarios_dir):
        # print(file)
        # if file.endswith(".py"):
            scenario_module_name = f"craftext.scenarios.{file}.instructions"
            scenario_module = importlib.import_module(scenario_module_name)
            scenarios.update(scenario_module.instructions)
    
    return scenarios

def get_random_instruction_and_checker():
    # Загружаем все сценарии
    all_scenarios = load_scenarios()
    print("Choose scnario:")
    for i in range(54, 58):#(len(all_scenarios)):
         print(f"{i}) {list(all_scenarios.keys())[i]}")
       #  print("Instruction example: ")
         scenario_key =list(all_scenarios.keys())[i]# random.choice(list(all_scenarios.keys()))
         scenario = all_scenarios[scenario_key]
         print("Instruction example: ", scenario['instruction'])

    idx = int(input("Input: "))
    # Случайно выбираем сценарий
    random_scenario_key =list(all_scenarios.keys())[idx]# random.choice(list(all_scenarios.keys()))
    random_scenario = all_scenarios[random_scenario_key]
    
    # Выбираем случайную инструкцию либо из 'instruction', либо из 'instruction_paraphrases'
    instructions = [random_scenario['instruction']] + random_scenario['instruction_paraphrases']
    random_instruction = instructions[0]#random.choice(instructions)
    keys = [0,0]
    # Возвращаем выбранную инструкцию и соответствующую check_lambda
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
        self.game_data.append(state)
        self.actions.append(action)
        # Добавляем инструкцию к наблюдению
        obs_with_instruction = [obs,self.instruction_keys]
       # print(self.current_instruction)
        done = self.checker(GameData(self.game_data,self.actions))
       # print(done)
        return obs_with_instruction, state, reward, done, info

if __name__=="__main__":
    instruction, checker = get_random_instruction_and_checker()
    print(instruction)