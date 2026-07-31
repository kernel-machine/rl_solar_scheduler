from stable_baselines3 import PPO
from lib.environment import EnvBeeDay, StateContent
from lib.solar.solar import Solar
import json
import numpy as np

with open("../runs/119/args.json") as f:
    args = json.load(f)

solar2025 = Solar("../solcast2025_solar_only.csv", scale_factor=2*0.55*0.51*0.1426, max_power=80, enable_cache=False)

test_env = EnvBeeDay(solar2025, 
    step_s=5*60,
    selected_day=1,
    start_hour=-1,
    end_hour=-1,
    acquistion_speed_fps=20,
    processing_speed_fps=30,
    seed=42, 
    state_content=StateContent.HOUR_MINUTE, 
    random_reset=True, 
    terminated_days=args["term_days"], 
    forecast_steps=args["forecast_steps"], 
    train_days=350,
    start_threshold=args["start_thr"],
    prevision_noise_amount=args["prevision_noise"],
    battery_weight=args["battery_weight"],
    buffer_weight=args["buffer_weight"],
    reward_shape=args["reward_shape"],
    battery_wh=args["battery_ah"]*12
)

model = PPO.load("../runs/119/last.pth", env=test_env, device='cpu')

obs, info = test_env.reset(options={"norandom":True, "day":0})
steps = 0
for i in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, terminated, info = test_env.step(action)
    print(f"Step: {i}, Action: {action.item():.2f}, Battery: {test_env.battery_curr_j:.0f}, Buffer: {test_env.buffer_length}, Done: {done}")
    if done:
        break
