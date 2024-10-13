import argparse
import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)
sys.path.append("./models")
from actor_critic import (ActorCriticConv, 
                          ActorCriticConvWithIdxEmbedding,
                          ActorCriticConvWithBERT)
from craftax.craftax_env import make_craftax_env_from_name
from craftext.craftext_wrapper import InstructionWrapper
sys.path.append(".")
from wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
)

def main(args):

    path_parts = os.path.normpath(args.path).split(os.sep)
    last_folder = path_parts[-2]  # Пред-последняя папка
    result_file = os.path.join(args.path, f"{last_folder}_checkpoint_results.txt")

    if not os.path.exists(result_file):
        open(result_file, 'w').close() 
        
    with open(os.path.join(args.path, "config.yaml")) as f:
        raw_config = yaml.load(f, Loader=yaml.Loader)

        config = {}
        for key, value in raw_config.items():
            if isinstance(value, dict) and "value" in value:
                config[key] = value["value"]

    config["NUM_ENVS"] = args.num_envs
    config["RATIO"] = args.ratio

    orbax_checkpointer = PyTreeCheckpointer()
    options = CheckpointManagerOptions(max_to_keep=1, create=True)
    checkpoint_manager = CheckpointManager(
        os.path.abspath(os.path.join(args.path, "policies")), orbax_checkpointer, options
    )

    is_classic = False
    config_name = config["ENV_NAME"]

    add_text_emb = "-Text" in config_name
    config["ENV_NAME"] = config_name.replace("-Text", "")
    
    env = make_craftax_env_from_name(config["ENV_NAME"],  False)
    actions_count = 17 if "Classic" in config["ENV_NAME"] else 43
    if "Pixels" in config["ENV_NAME"]:
            network = ActorCriticConvWithBERT(actions_count, config["LAYER_SIZE"])
    else:
            network = ActorCritic(actions_count, config["LAYER_SIZE"])
                                    

    env = InstructionWrapper(env, args.craftext_settings)
    env = OptimisticResetVecEnvWrapper(env, config["NUM_ENVS"], min(config["RATIO"],config["NUM_ENVS"]))
    env_params = env.default_params

    init_x = jnp.zeros((config["NUM_ENVS"], *env.observation_space(env_params).shape))

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng, __rng = jax.random.split(rng, 3)

    instructions = jnp.tile(env.encoded_instruction, (config["NUM_ENVS"], 1))

    network_params = network.init(_rng, init_x, instructions)

    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    train_state = checkpoint_manager.restore(config["TOTAL_TIMESTEPS"])

    obs, env_state = env.reset(_rng, env_params)
    done = False
    import time
    steps = 0
    step_fn = jax.jit(env.step)
    success_rate = jnp.zeros(config["NUM_ENVS"])
    total_success_rate = jnp.zeros(config["NUM_ENVS"])
    count_total_success_rate=jnp.zeros(config["NUM_ENVS"])
    done_count = jnp.zeros(config["NUM_ENVS"])
    while steps < 10000:
        pi, value = network.apply(train_state['params'], obs, env_state.instruction)
        action = pi.sample(seed=_rng)#[0]
        
        action = jax.device_put(action, device=jax.devices('gpu')[0])

        if action is not None:
            rng, _rng = jax.random.split(rng)
            obs, env_state, reward, done, info = step_fn(
                _rng, env_state, action, env_params
            )
            steps += 1
        
        instruction_done_float = info['SR']
        new_episode_sr = success_rate + instruction_done_float
        success_rate = new_episode_sr * (1 - done)
        total_success_rate=total_success_rate * (1 - done) + new_episode_sr * done 
        count_total_success_rate += new_episode_sr * done 
        done_count += jnp.float32(done)
    

    result_str = f"{args.craftext_settings} {args.num_envs} {np.mean(count_total_success_rate/done_count)}{np.std(count_total_success_rate/done_count)}\n"

    with open(result_file, 'a') as f:
        f.write(result_str)
    
   # print(np.mean(total_success_rate))
    print(np.mean(count_total_success_rate/done_count))
    print(np.std(count_total_success_rate/done_count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--craftext_settings", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--ratio", type=int, default=16)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.debug:
        with jax.disable_jit():
            main(args)
    else:
        main(args)
