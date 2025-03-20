import subprocess
from datetime import datetime

import numpy as np
import wandb
import os
from baselines.experiments.super_igor.super_dataset import SuperDataset
from faker import Faker

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
    


def run_policy_inference(llm_name, dataset_name, save_dataset_name, experiment_name, plan_with_llm, 
                         craftext_settings,num_return_sequences='5', augment="False"):
    args = [
            "python", "policy_inference.py", 
            "--experiment_name", experiment_name,
            "--craftext_settings", craftext_settings,
            "--num_envs", "1024",  
            "--plan_with_llm", str(plan_with_llm),
            "--inference", "1",  
            "--llm_path", llm_name,
            "--augment", str(augment),
            "--dataset_path", dataset_name,
            "--save_dataset_path", save_dataset_name,
            "--num_return_sequences", num_return_sequences,
        ]

    print(args)
    #exit()
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
    wandb.init(project="super_igor_cycle_rest")
    os.makedirs("super_experiments", exist_ok=True)
    
    #Make random name for expariment
    fake = Faker()

    name = fake.first_name()
    surname = fake.last_name()

    llm_name =  "./pretrained_plan_llm/3_" #""./external_experiments/_9"#"Qwen/Qwen2.5-3B-Instruct" 
    
    craftext_settings = "SI_simplified_set"
    use_llm_tuning = True
    start_from_checkpoint= False
    validate = False
    
    rl_done = 0
    inference_done = 0
    llm_done = 0
    
    rl_skip = 1 if start_from_checkpoint else 0
    inference_skip = 2 if start_from_checkpoint else 0
    llm_skip = 1 if start_from_checkpoint else 0
    
    if start_from_checkpoint:
        
        experiment_name = "super_experiments/simple_achivements_one_test_llmt_True_Lawrence_Gibbs_20250311_082123"
        temp_path = f"{experiment_name}/temp_dataset"
        dataset_name = f"{temp_path}/super_dataset{0}_{0}.json"
    else:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_tag = "simple_achivements_one_test"
        experiment_name = f"super_experiments/{experiment_tag}_llmt_{str(use_llm_tuning)}_{name}_{surname}_{current_time}"
        temp_path = f"{experiment_name}/temp_dataset"
        rl_experiment_path = "None"
        super_dataset_name = "None"
        os.makedirs(temp_path, exist_ok=True)
        
    for j in range(5):
        
        if rl_skip<= rl_done:
            dataset_name = f"{temp_path}/super_dataset{j}_{0}.json"
            run_policy_train(
                craftext_settings=craftext_settings,
                env_name="Craftax-Classic-Pixels-v1-Text",
                llm_path=llm_name,
                super_dataset=dataset_name,
                num_envs=1024,
                start_checkpoint_path=rl_experiment_path,
                experiment_name=experiment_name,
                encode_form_name="EMBED_CLS_FOR_SPLITS",
                total_timesteps=500000000
            )
        else:
            print("SKIP RL TRAINING!")
       # exit()
        rl_experiment_path = get_rl_checkpoint_path(experiment_name)
        rl_experiment_name = get_rl_experiment_name(rl_experiment_path)
        rl_done += 1
 
        for i in range(0,4):
            dataset_name = f"{temp_path}/super_dataset{j}_{i}.json"   
            output_dir = f'./{experiment_name}/llm_checkpoints/mix_text_cycle_{j}_{i}'
            plan_with_llm =   i>0 #Generate new plans after LLM training
            if inference_skip <= inference_done:
                augment = 1 if i<1 else 0
                save_dataset_path = dataset_name
                # Validation on train with new LLM and SuperDataset generation
                llm_checkpoint = llm_name+"/1_"
                run_policy_inference(llm_checkpoint,
                                    dataset_name, 
                                    save_dataset_path,
                                    experiment_name=rl_experiment_name,
                                    plan_with_llm=plan_with_llm,
                                    craftext_settings=craftext_settings,
                                    augment=augment,
                                    num_return_sequences='20')
                log_validation(dataset_name, context="train_dataset")
                #exit()
                save_dataset_path = f"{temp_path}/train_{j}_{i}.json"
                run_policy_inference(llm_checkpoint, save_dataset_path, save_dataset_path,
                                        plan_with_llm=True,
                                        experiment_name=rl_experiment_name,
                                        craftext_settings=craftext_settings, 
                                        augment=0,
                                        num_return_sequences='1')
                log_validation(save_dataset_path, context="train_1")
            else:
                print("SKIP RL INFERECNE!")
        
            inference_done += 1

           
            
            if i>0:
                # Merge datasets
                old_data_path = f"{temp_path}/super_dataset{j}_{i-1}.json"
                merger_last_datasets(old_data_path, dataset_name) 
                #
                # Validation with RL on parafrases and new goals
                if validate:
                    test_parafeases_results_path = f"{temp_path}/test_parafeases{j}_{i}.json"
                    run_policy_inference(llm_name, test_parafeases_results_path,
                                        experiment_name=rl_experiment_name,
                                        plan_with_llm=plan_with_llm,
                                        craftext_settings=craftext_settings+"_test_parafrases", 
                                        num_return_sequences='1')
                    log_validation(test_parafeases_results_path, context="test_parafeases")

                    test_new_obj_results_path = f"{temp_path}/test_new_obj{j}_{i}.json"
                    run_policy_inference(llm_name, test_new_obj_results_path, 
                                        experiment_name=rl_experiment_name,
                                        plan_with_llm=plan_with_llm,
                                        craftext_settings=craftext_settings+"_test_other_params", 
                                        num_return_sequences='1')
                    log_validation(test_new_obj_results_path, context="test_new_obj")

                # Train LLM
            if use_llm_tuning:
                if llm_skip <= llm_done:
                    llm_checkpoint = llm_name+"/1_"
                    run_llm_train(dataset_name, llm_checkpoint, output_dir)
                else:
                    print("SKIP LLM TRAIN!")
                llm_name = output_dir
                llm_done += 1
        super_dataset_name = dataset_name
