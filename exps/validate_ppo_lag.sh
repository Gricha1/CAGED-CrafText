cd ..
cd baselines
export COMET_API_KEY="3OfuYHwcRgIwG7DzgzJ190igY"

python ppo_lag_validation.py --craftext_settings achievements_safe_sequential_defeat_monster --env_name="Craftax-Classic-Pixels-v1-Text" --path /usr/home/workspace/baselines/checkpoints/PPO_LAG/exp_6