import os
import json
import glob 

def parse_instructions(file_name_txt):
    with open(file_name_txt, 'r') as file:
        content = file.read()
    # Split the content into separate dictionaries by '----'
    chunks = content.split('----')
    # Parse each chunk into a dictionary using eval
    print(chunks[0])
    instructions_list = []
    for chunk in chunks:
        try:
            if chunk.strip():
                c = eval(chunk.strip()) 
                instructions_list.append(c)
        except Exception as e:
            print(e)
            print(chunk)
   # instructions_list = [eval(chunk.strip()) for chunk in chunks if chunk.strip()]
    return instructions_list

def parse_instructions_from_folder(folder_path):
    instructions_list = []
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
    return previous_dict


# def process_instructions(data, train_file, test_file):
#     """
#     Processes a dictionary of instructions, grouping by `str_check_lambda`, and splits into two parts.

#     Args:
#         data (list): List of dictionaries with instruction data.
#         train_file (str): Path to the train file to save.
#         test_file (str): Path to the test file to save.

#     Returns:
#         None: Writes the results to train_file and test_file.
#     """
#     # Step 1: Group by `str_check_lambda`
#     grouped_instructions = {}
#     for item in data:
#         key = item["INSTRUCTION"]["str_check_lambda"]
#         if key not in grouped_instructions:
#             grouped_instructions[key] = {
#                 "INSTRUCTION": {
#                     "instruction": item["INSTRUCTION"]["instruction"],
#                     "instruction_paraphrases": item["INSTRUCTION"]["instruction_paraphrases"],
#                     "check_lambda": item["INSTRUCTION"]["str_check_lambda"],  # Use string representation
#                     "str_check_lambda": key
#                 }
#             }
#         else:
#             grouped_instructions[key]["INSTRUCTION"]["instruction_paraphrases"].extend(
#                 item["INSTRUCTION"]["instruction_paraphrases"]
#             )

#     # Convert grouped instructions to a list
#     grouped_list = list(grouped_instructions.values())

#     # Step 2: Split into two roughly equal parts
#     split_index = len(grouped_list) // 2
#     part1 = grouped_list[:split_index]
#     part2 = grouped_list[split_index:]

#     # Helper function to format JSON without quotes around `check_lambda`
#     def format_json(entry):
#         entry_json = json.dumps(entry, indent=4)
#         check_lambda_value = entry["INSTRUCTION"]["check_lambda"]
#         # Replace only the specific `check_lambda` field's quotes
#         entry_json = entry_json.replace(
#             f'"{check_lambda_value}"', check_lambda_value, 1
#         )
#         return entry_json

#     # Write to train and test files
#     with open(train_file, "w") as train_f:
#         for entry in part1:
#             train_f.write(format_json(entry))
#             train_f.write("\n----\n")

#     with open(test_file, "w") as test_f:
#         for entry in part2:
#             test_f.write(format_json(entry))
#             test_f.write("\n----\n")



# def train_test_split(folder_path, train_file="train.txt", test_file="test.txt"):
#     print(folder_path)
#     instructions_list = parse_instructions_from_folder(folder_path)
#     train_path = os.path.join(folder_path, train_file)
#     test_path = os.path.join(folder_path, test_file)
#     process_instructions(instructions_list,train_path, test_path)

# if __name__=="__main__":
#     jax_scenarious = glob.glob("./jax*")
#     #jax_scenarious = ["./jax_build_line"]
#     stop_list = ['jax_conditional_achivments']
#     for scenarious_path in jax_scenarious:
#         path = os.path.join(scenarious_path, "instructions/train/easy")
#         train_test_split(path, train_file="train.txt", test_file="test.txt")
#         path = os.path.join(scenarious_path, "instructions/train/medium")
#         train_test_split(path, train_file="train.txt", test_file="test.txt")
        