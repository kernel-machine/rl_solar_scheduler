import torch
import numpy as np
import argparse
from lib.environment import EnvBeeDay
from lib.solar.solar import Solar
from lib.solar.prediction import GHIPredictorTCN, GHIPredictorTransformer
from lib.utils import StateContent

parser = argparse.ArgumentParser()
args = parser.parse_args()

device = torch.device('cpu')
panel_area_m2 = 2*0.55*0.51
efficiency = 0.1426
max_power_w = 80
solar = Solar("../solcast2024_solar_only.csv", scale_factor=panel_area_m2*efficiency, max_power=max_power_w, enable_cache=False)

scaling_data = np.load('ghi_predictor_scaling.npz_tcn_scaling.npz')
nn_min_val = float(scaling_data['min_val'])
nn_max_val = float(scaling_data['max_val'])
nn_lookback = int(scaling_data['lookback'])
nn_horizon = int(scaling_data['horizon'])
nn_input_size = int(scaling_data['input_size'])
is_probabilistic = bool(scaling_data.get('probabilistic', False))
nn_use_log1p = bool(scaling_data.get('use_log1p', False))
output_size = nn_horizon * 3 if is_probabilistic else nn_horizon

model = GHIPredictorTCN(input_size=nn_input_size, hidden_size=64, num_layers=3, output_size=output_size)
model.load_state_dict(torch.load('ghi_predictor.pth_tcn.pth', map_location='cpu', weights_only=True))
model.eval()

env = EnvBeeDay(
    solar=solar,
    step_s=5*60,
    selected_day=1,
    start_hour=8,
    end_hour=18,
    acquistion_speed_fps=20,
    processing_speed_fps=20,
    state_content=StateContent.SOLAR | StateContent.NN_PREDICTION,
    nn_model=model,
    nn_lookback=nn_lookback,
    nn_horizon=nn_horizon,
    nn_min_val=nn_min_val,
    nn_max_val=nn_max_val,
    probabilistic_forecast=is_probabilistic,
    nn_use_log1p=nn_use_log1p
)

obs, _ = env.reset()
print("Observation size:", obs.shape)
print("Solar history length:", len(env.solar_history))
print("First history element:", env.solar_history[0])
print("Last history element:", env.solar_history[-1])

# Run a few steps to see predictions
for i in range(5):
    obs, reward, done, trunc, info = env.step(np.array([1.0]))
    fields = info['fields']
    nn_preds = [obs[idx] for idx, f in enumerate(fields) if 'NN Pred' in f]
    print(f"\nStep {i} NN Pred (first 5):", [round(x, 4) for x in nn_preds[:5]])
    print(f"Step {i} Real solar:", env.solar.get_solar_w(env.time_s))

