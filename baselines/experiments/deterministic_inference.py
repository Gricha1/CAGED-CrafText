import argparse
import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp
from functools import partial

from craftax.craftax_env import make_craftax_env_from_name
from craftext.craftext_wrapper import InstructionWrapper

class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)

class OptimisticResetVecEnvWrapper(GymnaxWrapper):
    """
    Provides efficient 'optimistic' resets.
    The wrapper also necessarily handles the batching of environment steps and resetting.
    reset_ratio: the number of environment workers per environment reset.  Higher means more efficient but a higher
    chance of duplicate resets.
    """

    def __init__(self, env, num_envs: int, reset_ratio: int):
        super().__init__(env)

        self.num_envs = num_envs
        self.reset_ratio = reset_ratio
        assert (
            num_envs % reset_ratio == 0
        ), "Reset ratio must perfectly divide num envs."
        self.num_resets = self.num_envs // reset_ratio

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs_st, state_st, reward, done, info = self.step_fn(rngs, state, action, params)

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_resets)
        obs_re, state_re = self.reset_fn(rngs, params)

        rng, _rng = jax.random.split(rng)
        reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)

        being_reset = jax.random.choice(
            _rng,
            jnp.arange(self.num_envs),
            shape=(self.num_resets,),
            p=done,
            replace=False,
        )
        reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))

        obs_re = obs_re[reset_indexes]
        state_re = jax.tree_map(lambda x: x[reset_indexes], state_re)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree_map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return state, obs

        state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


class MinimalExperiment:
    def __init__(self, args):
        self.args = args
        self.env = self._initialize_environment()

    def _initialize_environment(self):
        env_name = "Craftax-Classic-Pixels-v1"
        env = make_craftax_env_from_name(env_name, False)
        env = InstructionWrapper(env, self.args.craftext_settings)
        env = OptimisticResetVecEnvWrapper(env, self.args.num_envs, self.args.ratio) 
        return env

    def run(self):
        rng = jax.random.PRNGKey(42)
        n_actions = 17 # action space for Craftax-Classic

        obs, env_state = self.env.reset(rng, self.env.default_params)
        step_fn = jax.jit(self.env.step)
        steps = 0

        while steps < 5000:
            # Random action selection
            actions = jax.random.randint(rng, (obs.shape[0],), 0, n_actions)

            obs, env_state, reward, done, info = step_fn(rng, env_state, actions, self.env.default_params)
            print("INSTRUCTIONS: ", env_state.idx)
            print("RNGS: ", env_state.rng)
            steps += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--craftext_settings", type=str, default="simple")
    parser.add_argument("--num_envs", type=int, default=5, help="Number of environments")
    parser.add_argument("--ratio", type=int, default=1)

    args = parser.parse_args()

    experiment = MinimalExperiment(args)
    experiment.run()
