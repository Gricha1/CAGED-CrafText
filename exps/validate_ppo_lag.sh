cd ..
cd baselines
export COMET_API_KEY="3OfuYHwcRgIwG7DzgzJ190igY"

plotly_get_chrome -y

python ppo_lag_validation.py --craftext_settings achievements_safe_sequential_away_monsters_when_hp --env_name="Craftax-Classic-Pixels-v1-Text" \
                             --path /usr/home/workspace/baselines/checkpoints/PPO_LAG/exp_9