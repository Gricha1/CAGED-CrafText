import sys
import os
import argparse
import numpy as np
from baselines.experiments.super_igor.encoder import QwenEncodeModel
from baselines.experiments.super_igor.super_dataset import SuperDataset
import wandb
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_latest_folder(path='./results'):
    folders = [os.path.join(path, folder) for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
    if not folders:
        print("В папке нет подкаталогов.")
        return None
    latest_folder = max(folders, key=os.path.getctime)
    return latest_folder

def train_llm(dataset, 
              previos_checkpoint_dir="Qwen/Qwen2.5-3B-Instruct", 
              ouput_dir='./results'):
    
    llm_train_config = load_config('llama_training.yaml')
    llm_train_config['training_args']['output_dir'] = ouput_dir
    encoder = QwenEncodeModel(previos_checkpoint_dir)

    eos_token = encoder.tokenizer.eos_token
    llm_dataset, data_dict, bad_data_dict = dataset.llm_dataset(eos_token=eos_token)

    encoder.train(llm_dataset, bad_data_dict, llm_train_config, data_dict)

    return get_latest_folder(ouput_dir)


def main(dataset_name, llm_name, output_dir):
    dataset = SuperDataset.load_from_json(dataset_name)    
    new_llm_path = train_llm(dataset=dataset, 
                previos_checkpoint_dir=llm_name, 
                ouput_dir=output_dir)
    return new_llm_path

if __name__=="__main__":
    wandb.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default=None, type=str)
    parser.add_argument("--llm_name", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    args, rest_args = parser.parse_known_args(sys.argv[1:])
    main(args.dataset_name, args.llm_name, args.output_dir)
