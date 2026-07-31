from lib.environment import EnvBeeDay, StateContent
from lib.solar.solar import Solar
import numpy as np
import json

with open("../runs/113/args.json") as f:
    args = json.load(f)

solar2025 = Solar("../solcast2025_solar_only.csv", scale_factor=2*0.55*0.51*0.1426, max_power=80, enable_cache=False)

test_env = EnvBeeDay(solar2025, 
    step_s=5*60,
    selected_day=1,
    start_hour=-1 if args["autostart"] else 7,
    end_hour=-1,
    acquistion_speed_fps=20,
    processing_speed_fps=30,
    seed=42, 
    state_content=0, # doesn't matter for this
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

test_obs, info = test_env.reset(options={"norandom":True, "day":0})
print(f"Start hour: {test_env.start_hour}")
print(f"Initial battery: {test_env.battery_curr_j} J")

steps = 0
while True:
    obs, reward, done, terminated, info = test_env.step(np.array([1.0])) # Max action
    steps += 1
    if done:
        break
print(f"Died after {steps} steps")
print(f"Processed images: {test_env.processed_images}")
