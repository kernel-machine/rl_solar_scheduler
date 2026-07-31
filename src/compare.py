import torch
import numpy as np
from lib.environment import EnvBeeDay
from lib.solar.solar import Solar
from lib.solar.prediction import GHIPredictorLSTM
from lib.utils import StateContent

device = torch.device('cpu')
panel_area_m2 = 2*0.55*0.51
efficiency = 0.1426
max_power_w = 80
solar = Solar("../solcast2024_solar_only.csv", scale_factor=panel_area_m2*efficiency, max_power=max_power_w, enable_cache=False)

scaling_data = np.load('ghi_predictor_scaling.npz_lstm_scaling.npz')
nn_min_val = float(scaling_data['min_val'])
nn_max_val = float(scaling_data['max_val'])
nn_lookback = int(scaling_data['lookback'])
nn_horizon = int(scaling_data['horizon'])
nn_input_size = int(scaling_data['input_size'])

model = GHIPredictorLSTM(input_size=nn_input_size, hidden_size=64, num_layers=3, output_size=nn_horizon * 3)
model.load_state_dict(torch.load('ghi_predictor.pth_lstm.pth', map_location='cpu', weights_only=True))
model.eval()

env = EnvBeeDay(
    solar=solar,
    step_s=5*60,
    selected_day=10, 
    start_hour=10,
    end_hour=18,
    acquistion_speed_fps=20,
    processing_speed_fps=20,
    state_content=StateContent.SOLAR | StateContent.NN_PREDICTION | StateContent.SUN_REAL_PREDICTION,
    nn_model=model,
    nn_lookback=nn_lookback,
    nn_horizon=nn_horizon,
    nn_min_val=nn_min_val,
    nn_max_val=nn_max_val,
    probabilistic_forecast=True,
    nn_use_log1p=False
)
obs, info = env.reset()
fields = info['fields']

real_preds = [obs[idx] for idx, f in enumerate(fields) if 'Real Pred' in f]
nn_preds = [obs[idx] for idx, f in enumerate(fields) if 'NN Pred' in f]

print("Real:", [round(x, 4) for x in real_preds[:10]])
print("NN  :", [round(x, 4) for x in nn_preds[:10]])
diffs = [round(abs(r - n), 4) for r, n in zip(real_preds, nn_preds)]
print("Diffs:", diffs[:10])
print("Max Diff:", max(diffs))

