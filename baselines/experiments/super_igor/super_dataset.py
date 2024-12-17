import json
import numpy as np
from datasets import Dataset
from baselines.experiments.super_igor.prompts import promt_instruction


class Instruction:
    def __init__(self, instruction, plan_options, rewards=None):
        """
        Initialize an Instruction object.

        :param instruction: A string representing the instruction.
        :param plan_options: A list of strings, each representing a potential plan to achieve the instruction.
        :param rewards: A list of rewards corresponding to each plan option.
        """
        self.instruction = instruction
        self.plan_options = plan_options
        self.rewards = rewards if rewards is not None else [-1 for _ in range(len(plan_options))]

        self.mean_reward = 0
        self.update_mean_reward()
        self.sort_plans_by_reward()
    
    def update_mean_reward(self):
        np_rewards = np.array(self.rewards)
        mean_reward = np.mean(np_rewards[np_rewards>=0])
        self.mean_reward = mean_reward
        return
    
    def sort_plans_by_reward(self):
        paired_options = list(zip(self.rewards, self.plan_options))
        paired_options.sort(key=lambda x: x[0], reverse=True)
        self.rewards, self.plan_options = zip(*paired_options)
        self.rewards = list(self.rewards)[:5]
        self.plan_options = list(self.plan_options)[:5]
        return

    def map_instruction_to_plan(self):
        """
        Returns a dictionary where the instruction is the key,
        and the value is a list of tuples containing the plan options and their corresponding rewards.

        :return: dict {instruction: [(plan_option_1, reward_1), (plan_option_2, reward_2), ...]}
        """
        return {self.instruction: list(zip(self.plan_options, self.rewards))}
    
    def map_plan_to_instruction(self):
        """
        Returns a dictionary where each plan option is a key,
        and the value is a tuple of the corresponding instruction and reward.

        :return: dict {plan_option_1: (instruction, reward_1), ...}
        """
        return {plan: (self.instruction, reward) for plan, reward in zip(self.plan_options, self.rewards)}


    def update(self, plan_options, rewards):
        print(rewards)
        for i, plan_option in enumerate(plan_options):
            if rewards is None:
                reward_to_add = -1
            else:
                reward_to_add = rewards[i]
            if plan_option in self.plan_options:
                index = self.plan_options.index(plan_option)
                old_reward = self.rewards[index]
                new_reward = reward_to_add
                self.rewards[index] = self._update_reward(old_reward, new_reward)
            else:
                self.plan_options.append(plan_option)
                self.rewards.append(reward_to_add)
        self.update_mean_reward()
        self.sort_plans_by_reward()
 
    
    def _update_reward(self, old_reward, new_reward):
        if old_reward==-1:
            return new_reward
        return old_reward *0.1 + new_reward*0.9

    def to_dict(self):
        return {
            "instruction": self.instruction,
            "plan_options": self.plan_options,
            "rewards": self.rewards,
            "mean_reward": self.mean_reward
        }
    
    @staticmethod
    def from_dict(data):
        instruction = Instruction(
            instruction=data["instruction"],
            plan_options=data["plan_options"],
            rewards=data["rewards"]
        )
        return instruction
        

class SuperDataset:
    def __init__(self):
        self.instructions = dict()  # List of Instruction objects
        self.mapping_plan_to_instruction = dict()

    def add_instruction(self, instruction, plans, rewards = None):
        if instruction in self.instructions:
            self.instructions[instruction].update(plans, rewards)
        else:
            self.instructions[instruction] = Instruction(instruction=instruction, 
                                                         plan_options=plans, 
                                                         rewards=rewards)
        for plan in plans:
            self.mapping_plan_to_instruction[plan] = instruction
    
    def update(self, plan, reward):
        instruction = self.mapping_plan_to_instruction[plan]
        self.instructions[instruction].update([plan], [reward])
    
    def batch_update(self, plans, rewards):
        for plan, reward in zip(plans, rewards):
            #print(reward)
            self.update(plan, reward)

    def save_to_json(self, filepath):
        data = {
            "instructions": {key: instr.to_dict() for key, instr in self.instructions.items()},
            "mapping_plan_to_instruction": self.mapping_plan_to_instruction
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load_from_json(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        dataset = SuperDataset()
        dataset.instructions = {key: Instruction.from_dict(instr) for key, instr in data["instructions"].items()}
       # print(dataset.instructions)
        #exit()
        dataset.mapping_plan_to_instruction = data["mapping_plan_to_instruction"]
        return dataset
    
    def get_rewards(self, return_full=False):
        rewards = []
        for instruction in self.instructions.values():
            for i in range(len(instruction.rewards)):
                rewards.append(instruction.rewards[i])
        array = np.array(rewards)
        cleaned_array = array[~np.isnan(array)]
        sorted_array = np.sort(cleaned_array)  # Сортируем массив по возрастанию
        top_50_percent = sorted_array[int(len(sorted_array) / 2):]  # Берем 50% лучших значений
        if return_full:
            return top_50_percent, sorted_array
        return top_50_percent


    def _treshold(self):
        rewards = []
        for instruction in self.instructions.values():
            for i in range(len(instruction.rewards)):
                if instruction.rewards[i] > 0:  
                    rewards.append(instruction.rewards[i])
        
        array = np.array(rewards)
        cleaned_array = array[~np.isnan(array)]
        median = np.percentile(cleaned_array, 70) 
        return median

    @staticmethod
    def formatting(prompts, responses, eos_token):
        if eos_token:
            formatted_data = [f"{prompt}{response}\n {eos_token}" for prompt, response in zip(prompts, responses)]
        else:
            formatted_data = [f"{prompt}{response}\n" for prompt, response in zip(prompts, responses)]
        return formatted_data

    def to_dataset(self, fields=None):
        if fields is None:
            fields = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("_")]

        data_dict = {field: getattr(self, field) for field in fields}

        # Create a Hugging Face Dataset
        dataset = Dataset.from_dict(data_dict)
        return dataset

    def merge_and_optimize(self, other_dataset):
        for instruction, instr_obj in other_dataset.instructions.items():
            if instruction in self.instructions:
                current_instr = self.instructions[instruction]
                combined_plans = current_instr.plan_options + instr_obj.plan_options
                combined_rewards = current_instr.rewards + instr_obj.rewards
            else:
                combined_plans = instr_obj.plan_options
                combined_rewards = instr_obj.rewards
            if combined_rewards is not None:
                sorted_data = sorted(zip(combined_plans, combined_rewards), key=lambda x: x[1], reverse=True)
                top_5_plans, top_5_rewards = zip(*sorted_data[:5])
            else:
                top_5_plans = combined_plans[:5]
                top_5_rewards = [None] * len(top_5_plans)
            self.add_instruction(instruction, list(top_5_plans), list(top_5_rewards))


    def llm_dataset(self, eos_token=None):
        prompts = []
        answers = []
        treshold =  self._treshold()
        
        bad_examples_prompts = []
        bad_example_answer = []
        for instruction in self.instructions.keys():
            instruction_obj = self.instructions[instruction]
            for i in range(len(instruction_obj.rewards)):
                reward = instruction_obj.rewards[i]
                if reward >= treshold and reward is not np.nan:
                    prompts.append(promt_instruction(instruction))
                    answers.append(instruction_obj.plan_options[i])

                else:
                    bad_examples_prompts.append(promt_instruction(instruction))
                    bad_example_answer.append(instruction_obj.plan_options[i])
            
        text = SuperDataset.formatting(prompts, answers, eos_token)
        bad_text =  SuperDataset.formatting(bad_examples_prompts, bad_example_answer, eos_token)
        data_dict = {
        "text": text,
            }
        bad_data_dict = {
            "text":bad_text
            }
        full_dataset = Dataset.from_dict(data_dict)
        full_bad_dataset =  Dataset.from_dict(bad_data_dict)
    
        return full_dataset, data_dict, bad_data_dict
        


        
        