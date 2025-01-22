import os
from typing import List
### Imports for correct lambda-function parsing, dont remove! 
import jax.numpy as jnp

from craftext.checkers_jax.building import is_line_formed, is_square_formed
from craftext.checkers_jax.achivments import conditional_achivments
from craftext.checkers_jax.conditional import conditional_placing
from craftext.checkers_jax.relevant import place_object_relevant_to
from craftext.scenarios.constants import Achievement, MediumInventoryItems,InventoryItems,BlockType

def parse_instructions(file_name_txt: str) -> List[str]:
    with open(file_name_txt, 'r') as file:
        content = file.read()
    # Split the content into separate dictionaries by '----'
    chunks = content.split('----')
    # Parse each chunk into a dictionary using eval

    instructions_list = []
    for chunk in chunks:
        try:
            if chunk.strip():
                c = eval(chunk.strip()) 
                instructions_list.append(c)
        except Exception as e:
            pass

    return instructions_list

def parse_instructions_from_folder(folder_path):
    instructions_list = []
    print(os.listdir(folder_path))
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            instructions_list.extend(parse_instructions(file_path))
    return instructions_list

def update_previous_dict(previous_dict, folder_path, TASK_NAME='relevant_placement'):
    instructions_list = parse_instructions_from_folder(folder_path)

    for i, parsed_dict in enumerate(instructions_list):
        if 'INSTRUCTION' in parsed_dict:
            previous_dict[f"{TASK_NAME}_INSTRUCTION_{i}"] = parsed_dict['INSTRUCTION']
    

    print()
    print(len(previous_dict.keys()), folder_path)
    print()
    print("---"*30)
    return previous_dict
