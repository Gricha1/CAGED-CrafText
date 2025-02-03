import argparse
import os
import sys
import imageio
import jax
import json
import jax.numpy as jnp
import numpy as np
import pickle
import optax
import random
import yaml
import pandas as pd

from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

from baselines.models.actor_critic import (ActorCriticConv, 
                          ActorCriticConvWithIdxEmbedding,
                          ActorCriticConvWithBERT)

from baselines.wrappers import (
    OptimisticResetVecEnvWrapper,
)
from craftax.craftax_env import make_craftax_env_from_name

from craftext.craftext_wrapper import InstructionWrapper

from craftext.craftext_encoder import EncodeForm
from baselines.experiments.super_igor.encoder import QwenEncodeModel, QwenModelWrapper
from baselines.experiments.super_igor.scenarius_loader import CrafTextScenariosWithSuperDataset
from baselines.experiments.deterministic_inference import DetermOptimisticResetVecEnvWrapper
from baselines.experiments.super_igor.view import add_text_to_image, CraftaxRenderer
from baselines.experiments.super_igor.super_dataset import SuperDataset

class ResultManager:
    def __init__(self, experiment_name, craftext_settings):
        self.experiment_name = experiment_name
        self.craftext_settings = craftext_settings
        self.instructions = None
        self.functions = None
        self.success_rates = None
        self.std = None

    def update_results(self, instructions, functions, success_rates, std):
        self.instructions = instructions
        self.functions = functions
        self.success_rates = success_rates
        self.std = std

    def save_to_csv(self, output_path):
        if self.instructions is None or self.functions is None or self.success_rates is None:
            raise ValueError("Results are not initialized.")
        
        dataset = pd.DataFrame({
            'instructions': self.instructions,
            'functions': self.functions,
            'sr': self.success_rates,
            'std': self.std
        })
        dataset.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


class Experiment:
    def __init__(self, args):
        self.seeds_to_use = 50
        self.args = args
        self.config = self._load_config()
        self.checkpoint_manager = self._initialize_checkpoint_manager()
        self.env, self.network = self._initialize_environment_and_network()
        # Load pairs instriction - plans
        self.super_dataset = self.env.scenario_handler.super_dataset 
        self.train_state = self._initialize_train_state()
        
        self.result_manager = ResultManager(args.experiment_name, args.craftext_settings)

    def _load_config(self):
        config_path = os.path.join(self.args.path, "config.yaml")
        with open(config_path) as f:
            raw_config = yaml.load(f, Loader=yaml.Loader)

        config = {key: value["value"] if isinstance(value, dict) and "value" in value else value
                  for key, value in raw_config.items()}
        config["NUM_ENVS"] = self.args.num_envs
        config["RATIO"] = self.args.ratio
        config['INFERENCE'] = self.args.inference
        config['LLM_PATH'] = self.args.llm_path
        config['DATASET_PATH'] = self.args.dataset_path
        config['NUM_RETURN_SEQUENCES'] = self.args.num_return_sequences
        return config

    def _initialize_checkpoint_manager(self):
        orbax_checkpointer = PyTreeCheckpointer()
        options = CheckpointManagerOptions(max_to_keep=1, create=True)
        checkpoint_path = os.path.abspath(os.path.join(self.args.path, "checkpoint_restart_1"))
        return CheckpointManager(checkpoint_path, orbax_checkpointer, options)

    def _initialize_environment_and_network(self):
        is_classic = "-Text" not in self.config["ENV_NAME"]
        env_name = self.config["ENV_NAME"].replace("-Text", "")
        self.config["ENV_NAME"] = env_name

        env = make_craftax_env_from_name(env_name, False)
        actions_count = 17 if "Classic" in env_name else 43
        network_class = ActorCriticConvWithBERT if "Pixels" in env_name else ActorCriticConv
        network = network_class(actions_count, self.config["LAYER_SIZE"])

        EncodeModel = QwenModelWrapper(self.config["LLM_PATH"], num_return_sequences=self.config['NUM_RETURN_SEQUENCES'])
        env = InstructionWrapper(env, self.args.craftext_settings, scenario_handler_class=CrafTextScenariosWithSuperDataset,
                                  encode_model_class=EncodeModel,
                                  encode_form=EncodeForm.EMBED_CLS_FOR_SPLITS)
        
        print("N_INSTRUCTIOnS: ", env.n_instructions)
       # exit()
        if self.config['INFERENCE']:
            print("!-!")
            env = DetermOptimisticResetVecEnvWrapper(env, self.config["NUM_ENVS"], 
                                            min(self.config["RATIO"], self.config["NUM_ENVS"]),
                                            n_instructions=env.n_instructions, n_seeds=self.seeds_to_use)
        return env, network

    def _initialize_train_state(self):
        # init_x = jnp.zeros((self.config["NUM_ENVS"], *self.env.observation_space(self.env.default_params).shape))
        # rng = jax.random.PRNGKey(np.random.randint(2**31))
        # rng, _rng, __rng = jax.random.split(rng, 3)

        # instructions = jnp.tile(self.env.encoded_instruction, (self.config["NUM_ENVS"], 1))
        # network_params = self.network.init(_rng, init_x, instructions)

        # tx = optax.chain(
        #     optax.clip_by_global_norm(self.config["MAX_GRAD_NORM"]),
        #     optax.adam(self.config["LR"], eps=1e-5),
        # )
        # train_state = TrainState.create(
        #     apply_fn=self.network.apply,
        #     params=network_params,
        #     tx=tx,
        # )
        return self.checkpoint_manager.restore(int(self.config["TOTAL_TIMESTEPS"]))

    def view(self):
        rng = jax.random.PRNGKey(42)
        obs, env_state = self.env.reset(rng, self.env.default_params)
        step_fn = jax.jit(self.env.step)
        done = False
        renderer = CraftaxRenderer(self.env, self.env.default_params, pixel_render_size=1)
        steps = 0
        step_fn = jax.jit(self.env.step)
        params = self.train_state['runner_state'][0]["params"]
        observations = []
        while not done and steps < 5000:
            obs = jnp.expand_dims(obs, axis=0)
            instruction =self.env.scenario_handler.scenario_data.instructions_list[env_state.idx.item()]
            pi, value = self.network.apply(params, obs, env_state.instruction.reshape(1, -1))
            action = pi.sample(seed=rng)[0]
            
            action = jax.device_put(action, device=jax.devices('gpu')[0])

            if action is not None:
               # rng, _rng = jax.random.split(rng)
                obs, env_state, reward, done, info = step_fn(
                    rng, env_state, action, self.env.default_params
                )
                steps += 1

            image = renderer.render_to_image(env_state.env_state)
            observations.append(image)
        gif_name = "_".join(self.env.scenario_handler.scenario_data.instructions_list[env_state.idx.item()].split()[:5])
       # gif_name = inst.replace(" ", "_")
        ix = random.randint(0,200)
        folder_name = "animation"

        # Проверка и создание папки
        os.makedirs(folder_name, exist_ok=True)
        with imageio.get_writer(f'animation/{ix}_{gif_name}.gif', mode='I', duration=0.1) as writer:
            for i, image in enumerate(observations):
                text = f"Step {i}, Instruction {self.env.scenario_handler.scenario_data.instructions_list[env_state.idx.item()]}"
                image_with_text = add_text_to_image(image, text)
                writer.append_data(image_with_text.astype(np.uint8))
        print(f'Save with name animation/{ix}_{gif_name}.gif')

    def run(self):
        rng = jax.random.PRNGKey(42)
        
        SR = []

        for i in range(1):
            obs, env_state = self.env.reset(rng, self.env.default_params)
            step_fn = jax.jit(self.env.step)

            total_success_rate = np.zeros(self.config["NUM_ENVS"]*100)
            done_count = np.zeros(self.config["NUM_ENVS"]*100)
            prev_indx = np.zeros(self.config["NUM_ENVS"])
            prev_seeds = np.zeros(self.config["NUM_ENVS"])
            params = self.train_state['runner_state'][0]["params"]
            seeds_per_instruction = {}
            steps = 0
            remain_steps = 35000
            instr_rngs = dict()
            instr_rngs_alt = dict()
            steps_changed = False
            while steps < remain_steps:
                pi, value = self.network.apply(params, obs, env_state.env_state.instruction)
                action = pi.sample(seed=rng)
                if action is not None:
                    
                    obs, env_state, reward, done, info = step_fn(rng, env_state, action, self.env.default_params)
                    steps += 1
  
                    instruction_done_float = info['SR']

                    indices = np.where(instruction_done_float > 0)
                    done_indices = np.where(done > 0)
                    
                    inst_done = []
                    for inst in prev_indx[done_indices]:
                        if done_count[int(inst)] < self.seeds_to_use:
                            done_count[int(inst)] += 1
                        if done_count[int(inst)] == self.seeds_to_use:
                            inst_done.append(int(inst))
                        
                    for inst in prev_indx[indices]: 
                        if inst not in inst_done:
                            total_success_rate[int(inst)] += 1

                    
                        
                    prev_indx = env_state.env_state.idx
                    
                    
                    if np.sum(env_state.v3[1:])==0 and not steps_changed:
                        remain_steps = steps+300 # If all instructions_seeds pair starts, wait 300 steps for finish
                        steps_changed = True
                    else:
                        pass
                       # print(env_state.v3)
                        
                    indicec_used = np.array(env_state.env_state.idx)
                    rngs_used = np.array(env_state.env_state.rng)
                    for i, instruction in enumerate(indicec_used):
                        if instruction not in instr_rngs:
                            instr_rngs[int(instruction)] = []
                        instr_rngs[int(instruction)].append(tuple(rngs_used[i].tolist()))
                    
                    indicec_used = np.array(env_state.v1)
                    rngs_used = np.array(env_state.v2)
                    for i, instruction in enumerate(indicec_used):
                        if instruction not in instr_rngs_alt:
                            instr_rngs_alt[int(instruction)] = []
                        instr_rngs_alt[int(instruction)].append(tuple(rngs_used[i].tolist()))

            success_rates = total_success_rate / done_count
            SR.append(success_rates)
            
        # For seeds control
        for instruction in instr_rngs:
            instr_rngs[instruction] = tuple(set(instr_rngs[instruction]))
        
        for instruction in instr_rngs_alt:
            instr_rngs_alt[instruction] = tuple(set(instr_rngs_alt[instruction]))

        with open("seeds.json", "w", encoding="utf-8") as json_file:
            json.dump(instr_rngs, json_file, ensure_ascii=False, indent=4)
        
        with open("seeds_alt.json", "w", encoding="utf-8") as json_file:
            json.dump(instr_rngs_alt, json_file, ensure_ascii=False, indent=4)

        mean_sr = np.mean(np.array(SR), axis = 0)

        self.super_dataset.super_print()
        
        # FIX: Now is some problem in SR calulation: some env might be not fineshed, and one of that run many time on the same SEED
        self.super_dataset.batch_update(self.env.scenario_handler.scenario_data.instructions_list[1:], 
                                        mean_sr[1:len(self.env.scenario_handler.scenario_data.instructions_list)])
        self.super_dataset.save_to_json(self.config["DATASET_PATH"])

        print(total_success_rate[:10])
        print(done_count[:10])




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, type=str)
    parser.add_argument("--experiment_name", default=None, type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--craftext_settings", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--ratio", type=int, default=1)
    parser.add_argument("--inference", type=bool, default=False)
    parser.add_argument("--llm_path", type=str, default="Qwen/Qwen2.5-3B-Instruct-Advanced")
    parser.add_argument("--dataset_path", type=str, default="temp_dataset/super_dataset.json")
    parser.add_argument("--num_return_sequences", type=int, default=5)


    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if args.path is None:
        args.path = f"./wandb/{args.experiment_name}/files/"
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    experiment = Experiment(args)
    if args.debug:
        with jax.disable_jit():
            if not args.inference:
                experiment.view()
            else:
                experiment.run()
    else:
        if not args.inference:
                experiment.view()
        else:
                experiment.run()
