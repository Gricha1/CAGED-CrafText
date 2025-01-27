import subprocess
import numpy as np
import wandb
import os
from baselines.experiments.super_igor.super_dataset import SuperDataset


def run_policy_train(craftext_settings, env_name, 
                     llm_path, super_dataset, num_envs,
                     start_checkpoint_path=None, experiment_name="experiment", 
                     encode_form_name="EMBEDDING", total_timesteps=250000000):
    args = [
        "python", "policy_train.py",
        "--craftext_settings", craftext_settings,
        "--env_name", env_name,
        "--llm_path", llm_path,
        "--super_dataset", super_dataset,
        "--num_envs", str(num_envs),
        "--experiment_name", experiment_name,
        "--encode_form_name", encode_form_name,
        "--start_checkpoint_path", start_checkpoint_path,
        "--total_timesteps", str(total_timesteps)
    ]

    try:
        process = subprocess.Popen(args)
        process.wait()  # Дождаться завершения процесса
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Attempting to terminate the process...")
        process.terminate()  # Завершить процесс
        process.wait()  # Подождать завершения
    except Exception as e:
        print(f"Error occurred: {e}")
    except subprocess.CalledProcessError as e:
        print(f"RL Training failed with return code {e.returncode}")
        print(f"Error message: {e}")
        exit()
    finally:
        if process.poll() is None:
            print("Forcibly killing the process...")
            process.kill()
        print("Process terminated.")
    


def run_policy_inference(llm_name, dataset_name, experiment_name, 
                         craftext_settings,num_return_sequences='5'):
    args = [
            "python", "policy_inference.py", 
            "--experiment_name", experiment_name,
            "--craftext_settings", craftext_settings,
            "--num_envs", "256",  
            "--inference", "True",  
            "--llm_path", llm_name,
            "--dataset_path", dataset_name,
            "--num_return_sequences", num_return_sequences,
        ]

    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Inference failed with return code {e.returncode}")
        print(f"Error message: {e}")
        exit()

def run_llm_train(dataset_name, llm_name, output_dir):
    args = ["python", "llm_train.py",
        "--dataset_name", dataset_name,
        "--llm_name", llm_name,
        "--output_dir", output_dir]

    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"LLM Training failed with return code {e.returncode}")
        print(f"Error message: {e}")
        exit()



def log_validation(dataset_name, context=""):
    dataset = SuperDataset.load_from_json(dataset_name)
    top_50_percent, full =dataset.get_rewards(return_full=True)
    mean_rw = np.mean(top_50_percent)
    total_mean = np.mean(full)
    wandb.log({f"{context}_mean_top_50_sr":mean_rw})
    wandb.log({f"{context}_total_mean":total_mean})
    wandb.log({f"{context}_top_50_percent_sr": wandb.Histogram(top_50_percent)})

def merger_last_datasets(old_data_path, new_data_path):
    dataset = SuperDataset.load_from_json(old_data_path)
    dataset_new = SuperDataset.load_from_json(new_data_path)
    dataset.merge_and_optimize(dataset_new)
    dataset.save_to_json(new_data_path)

def get_rl_checkpoint_path(experiment_name):
    with open(f"{experiment_name}/path_to_last_checkpoint.txt", "r", encoding="utf-8") as file:
        content = file.read()
    return content

def get_rl_experiment_name(rl_experiment_path):
    p1 = rl_experiment_path.split("wandb/")[1]
    experiment_name = p1.split("/")[0]
    return experiment_name

if __name__=="__main__":
    run_policy_inference("Qwen/Qwen2.5-3B-Instruct" , "any.json",
                                    experiment_name="run-20250124_103954-5t1eqxol",
                                    craftext_settings="simple_achivments_one", 
                                    num_return_sequences='3')
    exit()
    

    wandb.init(project="super_igor_cycle_rest")

    
    llm_name = "Qwen/Qwen2.5-3B-Instruct" 
    experiment_name = "achivments_full_v6_test"
    temp_path = f"{experiment_name}/temp_dataset"
    craftext_settings = "simple_achivments"
    rl_experiment_path = "None"
    super_dataset_name = "None"
    
    os.makedirs(temp_path, exist_ok=True)

    for j in range(5):
        run_policy_train(
            craftext_settings=craftext_settings,
            env_name="Craftax-Classic-Pixels-v1-Text",
            llm_path=llm_name,
            super_dataset=super_dataset_name,
            num_envs=1024,
            start_checkpoint_path=rl_experiment_path,
            experiment_name=experiment_name,
            encode_form_name="EMBED_CLS_FOR_SPLITS",
            total_timesteps=250000000
        )
        
        rl_experiment_path = get_rl_checkpoint_path(experiment_name)
        rl_experiment_name = get_rl_experiment_name(rl_experiment_path)

        print("="*80)
        print()
        print(rl_experiment_path)
        print(rl_experiment_name)
        print()
        print("="*80)
        
       # rl_experiment_name='run-20241213_171349-clql9lvy'
        for i in range(0,4):
            dataset_name = f"{temp_path}/super_dataset{j}_{i}.json"
            output_dir = f'.{experiment_name}/llm_checkpoints/mix_text_cycle_{j}_{i}'
            
            # Validation on train with new LLM and SuperDataset generation
            run_policy_inference(llm_name, dataset_name, experiment_name=rl_experiment_name, craftext_settings=craftext_settings)
            log_validation(dataset_name, context="train_dataset")
           # exit()

            # Merge datasets
            if i>0:
                old_data_path = f"{temp_path}/super_dataset{j}_{i-1}.json"
                merger_last_datasets(old_data_path, dataset_name) 

                # Validation with RL on parafrases and new goals
                test_parafeases_results_path = f"{temp_path}/test_parafeases{j}_{i}.json"
                run_policy_inference(llm_name, test_parafeases_results_path,
                                    experiment_name=rl_experiment_name,
                                    craftext_settings="simple_achivments_test_parafrases", 
                                    num_return_sequences='1')
                log_validation(test_parafeases_results_path, context="test_parafeases")

                test_new_obj_results_path = f"{temp_path}/test_new_obj{j}_{i}.json"
                run_policy_inference(llm_name, test_new_obj_results_path, 
                                    experiment_name=rl_experiment_name,
                                    craftext_settings="simple_achivments_test_other_params", 
                                    num_return_sequences='1')
                log_validation(test_new_obj_results_path, context="test_new_obj")

                # Train LLM
            run_llm_train(dataset_name, llm_name, output_dir)
            llm_name = output_dir 
        super_dataset_name = dataset_name
