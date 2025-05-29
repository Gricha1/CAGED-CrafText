import wandb
import argparse
import os
import sys
import textwrap
from PIL import Image, ImageDraw, ImageFont
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
import pygame
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from craftax.craftax.renderer import render_craftax_pixels as render
from craftax.craftax_classic.renderer import render_craftax_pixels as render_classic
from craftax.craftax.constants import (
    OBS_DIM,
    BLOCK_PIXEL_SIZE_HUMAN,
    INVENTORY_OBS_HEIGHT,
    Action,
    Achievement,
)
from craftext.environment.craftext_wrapper_cmdp import CMDPInstructionWrapper
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)
import orbax.checkpoint as ocp
sys.path.append("./models")
from baselines.models.actor_critic import ActorCriticConv
from models.actor_critic_with_text_constraints import (
    ActorCriticConvWithBERTCMDP
)
from models.actor_critic_with_text import (
    ActorCriticConvWithBERT
)
import imageio
from craftax.craftax_env import make_craftax_env_from_name

os.environ["SDL_VIDEODRIVER"] = "dummy"

try:
    font = ImageFont.truetype("arial.ttf", 30)
except IOError:
    font = ImageFont.load_default()

# Установим стиль
sns.set(style="whitegrid")
colors = {
    'reward': '#A3D2CA',       # мягкий бирюзовый
    'cost': '#F6BD60',         # теплый желто-оранжевый
    'success_rate': '#84A59D'  # спокойный серо-зеленый
}


def add_text_to_image(image, text):
    """Add text to an image."""
    text_to_list = text.split()
    text = ""
    for i in range(0, len(text_to_list), 6):
        text += " ".join(text_to_list[i:i+6])
        text += "\n"
        
    img_pil = Image.fromarray(image.astype(np.uint8))
    img_with_text = Image.new('RGB', (img_pil.width, img_pil.height + 50), color=(255, 255, 255))
    img_with_text.paste(img_pil, (0, 50))

    draw = ImageDraw.Draw(img_with_text)
    draw.text((10, 5), text, font=font, fill=(0, 0, 0))

    return np.array(img_with_text)


from PIL import Image

class CraftaxRenderer:
    def __init__(self, env, env_params, pixel_render_size=4):
        self.env = env
        self.env_params = env_params
        self.pixel_render_size = pixel_render_size
        self.frames = []

        self.screen_size = (
            OBS_DIM[1] * BLOCK_PIXEL_SIZE_HUMAN * pixel_render_size,
            (OBS_DIM[0] + INVENTORY_OBS_HEIGHT) * BLOCK_PIXEL_SIZE_HUMAN * pixel_render_size,
        )
        if self.env.environment_key == 1:
            env_render = render
        else:
            env_render = render_classic
        self._render = jax.jit(env_render, static_argnums=(1,))

    def render(self, env_state):
        pixels = self._render(env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=0)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=1)

        image = Image.fromarray(np.array(pixels).astype(np.uint8))
        self.frames.append(image)

    def render_to_image(self, env_state):
        """Render the environment state to an image array and resize it to 256x256."""
        pixels = self._render(env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=0)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=1)
        
        # Convert pixels to image and resize to 256x256
        image = Image.fromarray(np.array(pixels).astype(np.uint8))
        resized_image = image.resize((256, 256))
        
        return np.array(resized_image)

    def save_gif(self, filename, duration=100):
        """Save the stored frames as a GIF."""
        if self.frames:
            self.frames[0].save(
                filename,
                save_all=True,
                append_images=self.frames[1:],
                duration=duration,
                loop=0
            )


def main(args):
    
    wandb.init(project="craftext_ppo_validation", name="validation_ppo", mode="online")  # или mode="disabled" для оффлайн

    # folder for animation
    animation_dir = "animation"
    os.makedirs(animation_dir, exist_ok=True)
    for filename in os.listdir(animation_dir):
        file_path = os.path.join(animation_dir, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    def find_latest_checkpoint(base_path):
        # Получаем список всех подпапок в базовом пути
        try:
            subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        except FileNotFoundError:
            raise ValueError(f"Папка {base_path} не существует")
        
        # Фильтруем только числовые папки
        numeric_folders = []
        for folder in subfolders:
            try:
                num = int(folder)
                numeric_folders.append(num)
            except ValueError:
                continue
        
        if not numeric_folders:
            raise ValueError(f"В папке {base_path} нет подпапок с числовыми именами")
        
        # Находим максимальное число
        latest_num = max(numeric_folders)
        
        # Формируем полный путь
        full_path = os.path.join(base_path, str(latest_num), "default")
        
        return full_path

    #checkpoint_path = args.path  # Пример: checkpoints/PPO_LAG/exp_1
    checkpoint_path = find_latest_checkpoint(args.path)
    print(f"download weights path: {checkpoint_path}")

    # Восстанавливаем веса
    checkpointer = PyTreeCheckpointer()
    restored = checkpointer.restore(checkpoint_path)

    train_state: TrainState = restored["train_state"]

    config = {}
    config["NUM_ENVS"] = 1
    config["ENV_NAME"] = args.env_name

    is_pixels = "Pixels" in config["ENV_NAME"]
    actions_count = 17 if "Classic" in config["ENV_NAME"] else 43

    env_name = config["ENV_NAME"].replace("-Text", "")
    env = make_craftax_env_from_name(env_name, False)
    env = CMDPInstructionWrapper(env, args.craftext_settings)
    env_params = env.default_params

    network = ActorCriticConvWithBERTCMDP(
            env.action_space(env_params).n, 512)

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng, __rng = jax.random.split(rng, 3)

    init_x = jnp.zeros((config["NUM_ENVS"], *env.observation_space(env_params).shape))

    encoded_instruction = jnp.expand_dims(env.encoded_instruction, axis=0) 
    encoded_constraint = jnp.expand_dims(env.encoded_textual_constraint, axis=0) 
    network_params = network.init(_rng, init_x, encoded_instruction, encoded_constraint)

    num_tasks = len(env.scenario_handler.scenario_data_jax.constraints_embeddings_list)
    tasks_ids = list(range(0, num_tasks))
    #tasks_ids = list(range(0, 2))
    print("tasks num:", len(env.scenario_handler.scenario_data_jax.constraints_embeddings_list))
    task_metrics = []

    for task_id in tasks_ids:
        print()
        print("Validation task id:", task_id, " of", num_tasks - 1)
        obs, env_state = env.reset(_rng, env_params, instruction_idx=task_id)
        done = False

        renderer = CraftaxRenderer(env, env_params, pixel_render_size=1)
        steps = 0
        step_fn = jax.jit(env.step, static_argnums=3)
        observations = []
        return_reward = 0
        return_cost = 0
        while not done and steps < 500:
            obs = jnp.expand_dims(obs, axis=0)
            instruction = env.scenario_handler.scenario_data.instructions_list[env_state.idx]
            textual_constraint = env.scenario_handler.scenario_data.texutal_constraints_list[env_state.idx]
            pi, value, cost_value = network.apply(train_state['params'], obs, 
                                    env_state.instruction.reshape(1, -1),
                                    env_state.textual_constraint.reshape(1, -1))
            action = pi.sample(seed=_rng)[0]
            
            action = jax.device_put(action, device=jax.devices('gpu')[0])

            if action is not None:
                rng, _rng = jax.random.split(rng)
                obs, env_state, reward, done, info = step_fn(
                    _rng, env_state, action, env_params
                )
                steps += 1
                
                return_reward += reward
                return_cost += env_state.cost

            image = renderer.render_to_image(env_state.env_state)
            observations.append(image)

        # get metrics
        total_success_rate = env_state.total_success_rate
        task_metrics.append({
            "task_id": task_id,
            "instruction": instruction,
            "constraint": textual_constraint,
            "reward": float(return_reward),
            "cost": float(return_cost),
            "success_rate": float(total_success_rate)
        })

        # get gif
        gif_name = instruction.replace(" ", "_")
        import random
        ix = random.randint(0,200)
        with imageio.get_writer(f'animation/{ix}_{gif_name}_{task_id}.gif', mode='I', duration=0.1) as writer:
            for i, image in enumerate(observations):
                text = f"Step {i}, Instruction: {instruction} \n Constrain: {textual_constraint}"
                image_with_text = add_text_to_image(image, text)
                writer.append_data(image_with_text.astype(np.uint8))

        wandb.log({
            "animation": wandb.Video(f"animation/{ix}_{gif_name}_{task_id}.gif", fps=10, format="gif")
        })


    # Texts
    fig, ax = plt.subplots(figsize=(12, len(task_metrics)*1.2))
    ax.axis('off')

    text_lines = []
    for i in range(len(task_metrics)):
        instr_wrapped = "\n".join(textwrap.wrap(task_metrics[i]['instruction'], width=70))
        constraint_wrapped = "\n".join(textwrap.wrap(task_metrics[i]['constraint'], width=70))
        text_lines.append(
            f"Task {task_metrics[i]['task_id']}:\nInstruction: {instr_wrapped}\nConstraint: {constraint_wrapped}\n"
        )

    full_text = "\n\n".join(text_lines)
    ax.text(0, 1, full_text, fontsize=9, verticalalignment='top', family='monospace')

    text_path = "task_descriptions.png"
    plt.tight_layout()
    plt.savefig(text_path, dpi=150)
    plt.close()

    wandb.log({
        "task_descriptions": wandb.Image(text_path)
    })

    # Bars
    x = list(range(len(task_metrics)))
    rewards = [item['reward'] for item in task_metrics]
    costs = [item['cost'] for item in task_metrics]
    success_rates = [item['success_rate'] for item in task_metrics]

    # Barplot
    fig, ax1 = plt.subplots(figsize=(max(12, len(x) * 0.7), 6))
    bar_width = 0.25

    # Отрисовка баров
    ax1.bar([i - bar_width for i in x], rewards, width=bar_width, label='Reward', color=colors['reward'])
    ax1.bar(x, costs, width=bar_width, label='Cost', color=colors['cost'])
    ax1.bar([i + bar_width for i in x], success_rates, width=bar_width, label='Success Rate', color=colors['success_rate'])

    # Подписи над столбцами
    for i in range(len(x)):
        ax1.text(i - bar_width, rewards[i] + 0.05, f"{rewards[i]:.1f}", ha='center', fontsize=8)
        ax1.text(i, costs[i] + 0.05, f"{costs[i]:.1f}", ha='center', fontsize=8)
        ax1.text(i + bar_width, success_rates[i] + 0.05, f"{success_rates[i]:.1f}", ha='center', fontsize=8)

    # Настройка осей
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(item['task_id']) for item in task_metrics], rotation=0)
    ax1.set_ylabel("Value")
    ax1.set_xlabel("Task ID")
    ax1.set_title("Task Metrics Overview")
    ax1.legend()
    plt.tight_layout()

    # Сохраняем
    plot_path = "task_metrics_barplot.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    wandb.log({
        "task_metrics_barplot": wandb.Image(plot_path)
    })


def print_new_achievements(achievements_cls, old_achievements, new_achievements):
    for i in range(len(old_achievements)):
        if old_achievements[i] == 0 and new_achievements[i] == 1:
            print(f"{achievements_cls(i).name} ({new_achievements.sum()}/{22})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--craftext_settings", type=str, default=None)
    parser.add_argument("--env_name", type=str, default="Craftax-Pixels-v1-Text")

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.debug:
        with jax.disable_jit():
            main(args)
    else:
        main(args)