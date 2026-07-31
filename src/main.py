from stable_baselines3 import PPO,A2C,DQN
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common import logger as sb3_logger
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from lib.environment import EnvBeeDay
from lib.utils import StateContent
import matplotlib.pyplot as plt
import argparse
import datetime
from datetime import timedelta
import os
import json
import torch
import numpy as np
import random
from stable_baselines3.common.monitor import Monitor
from functools import partial
import shutil
from sb3_contrib import RecurrentPPO
from utils import *
from lib.solar.solar import Solar
from statistics import mean, stdev
import signal
import sys
from tqdm import tqdm

from stable_baselines3.common.vec_env import SubprocVecEnv
from time import time
from thop import profile, clever_format

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",default=10000, required=False, type=int)
    parser.add_argument("--update_steps",default=512, required=False, type=int)
    parser.add_argument("--epochs",default=5, required=False, type=int)
    parser.add_argument("--alg", default="ppo", choices=["ppo","a2c","dqn","rec_ppo"])
    parser.add_argument("--use_solar", default=False, action="store_true")
    parser.add_argument("--use_month", default=False, action="store_true")
    parser.add_argument("--use_hour", default=False, action="store_true")
    parser.add_argument("--use_hour_minute", default=False, action="store_true")
    parser.add_argument("--use_minute", default=False, action="store_true")
    parser.add_argument("--use_day", default=False, action="store_true")
    parser.add_argument("--use_day_avg", default=False, action="store_true")
    parser.add_argument("--use_next_day", default=False, action="store_true")
    parser.add_argument("--use_real_forecast", default=False, action="store_true")
    parser.add_argument("--use_estimate_forecast", default=False, action="store_true")
    parser.add_argument("--use_estimate_single_forecast", default=False, action="store_true")
    parser.add_argument("--use_humidity", default=False, action="store_true")
    parser.add_argument("--use_cloud", default=False, action="store_true")
    parser.add_argument("--use_pressure", default=False, action="store_true")
    parser.add_argument("--use_sunset_time", default=False, action="store_true")
    parser.add_argument("--use_embed_day", default=False, action="store_true")
    parser.add_argument("--use_embed_next_day", default=False, action="store_true")
    parser.add_argument("--use_embed_prev_day", default=False, action="store_true")
    parser.add_argument("--use_quantize_day", default=False, action="store_true")
    parser.add_argument("--use_quantize_prev_day", default=False, action="store_true")
    parser.add_argument("--use_solar_horizon", default=False, action="store_true")
    parser.add_argument("--layer_width", default=64, type=int)
    parser.add_argument("--layer_depth", default=2, type=int)
    parser.add_argument("--latent_size", default=24, type=int)
    parser.add_argument("--run_folder", default="../runs", type=str)
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--val",default=False, action="store_true")
    parser.add_argument("--model",default=None, type=str)
    parser.add_argument("--lr",type=float, default=0.0003)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--device_idle_energy_w", type=float, default=2.5)
    parser.add_argument("--device_full_energy_w", type=float, default=7.5)
    parser.add_argument("--lr_decay",default=None, choices=["exp","cos","lin","lin_half","lin_thirth"])
    parser.add_argument("--n_env", default=1, type=int, required=False)
    parser.add_argument("--term_days", default=1, type=int, required=False)
    parser.add_argument("--forecast_steps",type=int, default=128)
    parser.add_argument("--prediction_accuracy", default=1.0, type=float)
    parser.add_argument("--choose_forecast",default=False, action="store_true")
    parser.add_argument("--test_year",type=int, default=2025)
    parser.add_argument("--test_freq",type=int, default=0)
    parser.add_argument("--train_days", type=int, default=30)
    parser.add_argument("--autostart", default=False, action="store_true")
    parser.add_argument("--start_thr",type=float, default=0.1)
    parser.add_argument("--prevision_noise", type=float, default=0.1)
    parser.add_argument("--use_images", default=False, action="store_true")
    parser.add_argument("--only_day_acquisition", default=False, action="store_true")
    parser.add_argument("--battery_weight", type=float, default=1.0)
    parser.add_argument("--buffer_weight", type=float, default=1.0)
    parser.add_argument("--random_day_switch", default=False, action="store_true")
    parser.add_argument("--discrete_action", default=False, action="store_true")
    parser.add_argument("--reward_shape", type=int, default=1)
    parser.add_argument("--buffer_incoming", default=False, action="store_true")
    parser.add_argument("--lstm_prediction", default=False, action="store_true", help="Enable LSTM solar prediction in observations")
    parser.add_argument("--lstm_model", type=str, default="ghi_predictor_lstm.pth", help="Path to pre-trained LSTM model")
    parser.add_argument("--tcn_prediction", default=False, action="store_true", help="Enable TCN solar prediction in observations")
    parser.add_argument("--tcn_model", type=str, default="ghi_predictor_tcn.pth", help="Path to pre-trained TCN model")
    parser.add_argument("--transformer_prediction", default=False, action="store_true", help="Enable Transformer solar prediction in observations")
    parser.add_argument("--transformer_model", type=str, default="ghi_predictor_transformer.pth", help="Path to pre-trained Transformer model")
    parser.add_argument("--probabilistic_forecast", default=False, action="store_true", help="Use probabilistic forecast (3 values per step)")
    parser.add_argument("--nn_lookback", type=int, default=96, help="NN lookback window size")
    parser.add_argument("--nn_horizon", type=int, default=24, help="NN prediction horizon")
    parser.add_argument("--battery_ah", type=float, default=24.0, help="Battery capacity in ampere-hours")
    args = parser.parse_args()
    SEED = 42

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    set_random_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Should fix type 3 font error
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    if not args.val and args.run_folder == "../runs":
        if not os.path.exists(args.run_folder):
            os.mkdir(args.run_folder)
        folders = os.listdir(args.run_folder)
        folders = list(filter(lambda x:x.isdigit(), folders))
        if len(folders)==0:
            folders=[0]
        last_id = max(map(lambda x:int(x),folders))
        new_id = last_id+1
        args.run_folder = os.path.join(args.run_folder,str(new_id))
        
        def signal_handler(sig, frame):
            shutil.rmtree(args.run_folder)
            print(f"Run folder {args.run_folder} cleaned")
            sys.exit(0)
        signal.signal(signal.SIGTERM, signal_handler) #SIGTEM sent from tsp -k

    if args.run_folder is not None:
        if os.path.exists(args.run_folder) and not args.val:
            print("Run folder already exists!")
            exit(0)
        else:
            os.makedirs(args.run_folder, exist_ok=True)
            print("Files were saved on",args.run_folder)

    if args.run_folder is not None and not args.val:
        file = os.path.join(args.run_folder,"args.json")
        with open(file, 'w', encoding='utf-8') as f:
            j = json.dumps(args.__dict__, ensure_ascii=False, indent=4)
            f.write(j)

    state_content = 0
    if args.use_solar:
        state_content ^= StateContent.SOLAR
    if args.use_month:
        state_content ^= StateContent.MONTH
    if args.use_hour:
        state_content ^= StateContent.HOUR
    if args.use_day:
        state_content ^= StateContent.DAY
    if args.use_next_day:
        state_content ^= StateContent.NEXT_DAY
    if args.use_real_forecast:
        state_content ^= StateContent.SUN_REAL_PREDICTION
    if args.use_estimate_forecast:
        state_content ^= StateContent.SUN_ESTIMATE_PREDICTION
    if args.use_estimate_single_forecast:
        state_content ^= StateContent.SUN_ESTIMATE_SINGLE_PREDICTION
    if args.use_pressure:
        state_content ^= StateContent.PRESSURE
    if args.use_cloud:
        state_content ^= StateContent.CLOUD
    if args.use_humidity:
        state_content ^= StateContent.HUMIDITY
    if args.use_minute:
        state_content ^= StateContent.MINUTE
    if args.use_hour_minute:
        state_content ^= StateContent.HOUR_MINUTE
    if args.use_sunset_time:
        state_content ^= StateContent.SUNSET_TIME
    if args.use_day_avg:
        state_content ^= StateContent.DAY_AVG
    if args.use_embed_day:
        state_content ^= StateContent.EMBEDDED_CURRENT_DAY
    if args.use_quantize_day:
        state_content ^= StateContent.QUANTIZED_DAY
    if args.use_quantize_prev_day:
        state_content ^= StateContent.QUANTIZED_PREV_DAY
    if args.use_embed_next_day:
        state_content ^= StateContent.EMBEDDED_NEXT_DAY
    if args.use_embed_prev_day:
        state_content ^= StateContent.EMBEDDED_PREV_NEXT_DAY
    if args.use_images:
        state_content ^= StateContent.IMAGES
    if args.use_solar_horizon:
        state_content ^= StateContent.SOLAR_HORIZON
    if args.lstm_prediction or args.tcn_prediction or args.transformer_prediction:
        state_content ^= StateContent.NN_PREDICTION

    state_content ^= StateContent.BUFFER

    device = torch.device("cuda:0") if args.gpu else torch.device("cpu")

    panel_area_m2 = 2*0.55*0.51 #m2
    efficiency = 0.1426
    max_power_w = 80 #W
    battery_v = 12
    battery_ah = args.battery_ah
    battery_wh = battery_ah*battery_v
    solar2024 = Solar("../solcast2024_solar_only.csv", scale_factor=panel_area_m2*efficiency, max_power=max_power_w, enable_cache=True, prediction_accuracy=args.prediction_accuracy)
    solar2025 = Solar(f"../solcast{args.test_year}_solar_only.csv", scale_factor=panel_area_m2*efficiency, max_power=max_power_w, enable_cache=True, prediction_accuracy=args.prediction_accuracy)

    # Load NN model if needed
    nn_model_instance = None
    nn_min_val = 0.0
    nn_max_val = 1.0
    nn_use_log1p = False
    args.nn_prediction = args.lstm_prediction or args.tcn_prediction or args.transformer_prediction
    if args.nn_prediction:
        if args.lstm_prediction:
            from lib.solar.prediction import GHIPredictorLSTM
            model_path = args.lstm_model
            model_class = GHIPredictorLSTM
        elif args.tcn_prediction:
            from lib.solar.prediction import GHIPredictorTCN
            model_path = args.tcn_model
            model_class = GHIPredictorTCN
        else:
            from lib.solar.prediction import GHIPredictorTransformer
            model_path = args.transformer_model
            model_class = GHIPredictorTransformer

        scaling_path = model_path.replace('.pth', '_scaling.npz')
        if os.path.exists(scaling_path):
            scaling_data = np.load(scaling_path, allow_pickle=True)
            nn_min_val = float(scaling_data['min_val'])
            nn_max_val = float(scaling_data['max_val'])
            nn_hidden_size = int(scaling_data['hidden_size'])
            nn_num_layers = int(scaling_data['num_layers'])
            nn_horizon = int(scaling_data['horizon'])
            nn_lookback = int(scaling_data['lookback'])
            nn_input_size = int(scaling_data['input_size']) if 'input_size' in scaling_data else 3
            is_probabilistic = bool(scaling_data.get('probabilistic', False)) or args.probabilistic_forecast
            nn_use_log1p = bool(scaling_data.get('use_log1p', False))
            args.nn_horizon = nn_horizon
            args.nn_lookback = nn_lookback
            print(f"NN scaling: min={nn_min_val:.2f}, max={nn_max_val:.2f}, lookback={nn_lookback}, horizon={nn_horizon}, input_size={nn_input_size}, probabilistic={is_probabilistic}, log1p={nn_use_log1p}")
        else:
            nn_hidden_size = 64
            nn_num_layers = 3
            nn_input_size = 3
            is_probabilistic = args.probabilistic_forecast
            nn_use_log1p = False
            print(f"Warning: scaling file {scaling_path} not found, using default parameters")
        
        model_output_size = args.nn_horizon * 3 if is_probabilistic else args.nn_horizon
        args.forecast_steps_nn = args.nn_horizon # To pass to feature extractor (always horizon size now)
        args.probabilistic_forecast = is_probabilistic

        nn_model_instance = model_class(
            input_size=nn_input_size, 
            hidden_size=nn_hidden_size, 
            num_layers=nn_num_layers, 
            output_size=model_output_size
        )
        nn_model_instance.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        nn_model_instance.eval()
        nn_model_instance.to(device)
        print(f"NN model loaded from {model_path}")

    start_hour = -1 if args.autostart else 7
    end_hour = -1 if args.autostart else 18
    def make_env_factory(env_id: int):
        def env_create():
            print("ENV ID",env_id)
            env = EnvBeeDay(solar2024, 
                step_s=5*60,
                selected_day=1,
                start_hour=start_hour,
                end_hour=end_hour,
                acquistion_speed_fps=20,
                processing_speed_fps=30,
                seed=SEED, 
                state_content=state_content, 
                random_reset=True, 
                terminated_days=args.term_days, 
                forecast_steps=args.forecast_steps, 
                choose_forecast=args.choose_forecast,
                latent_size=args.latent_size,
                train_days=args.train_days,
                start_threshold=args.start_thr,
                prevision_noise_amount=args.prevision_noise,
                battery_wh=battery_wh,
                only_day_acquisition=args.only_day_acquisition,
                battery_weight=args.battery_weight,
                device_idle_energy_w=args.device_idle_energy_w,
                device_full_energy_w=args.device_full_energy_w,
                buffer_weight=args.buffer_weight,
                random_day_switch=args.random_day_switch,
                discrete_action=args.discrete_action,
                reward_shape=args.reward_shape,
                buffer_incoming=args.buffer_incoming,
                nn_model=nn_model_instance,
                nn_lookback=args.nn_lookback,
                nn_horizon=args.nn_horizon,
                nn_min_val=nn_min_val,
                nn_max_val=nn_max_val,
                probabilistic_forecast=args.probabilistic_forecast,
                nn_use_log1p=nn_use_log1p)
            if env_id == 0:
                log_path = os.path.join(args.run_folder, f"monitor_{env_id}")
                env = Monitor(env, log_path)
            return env
        return env_create
    env_fns = [make_env_factory(i) for i in range(args.n_env)]
    vec_env = SubprocVecEnv(env_fns)
    #vec_env = make_vec_env(env_create, n_envs=args.n_env, vec_env_cls=SubprocVecEnv, seed=SEED)

    test_env = EnvBeeDay(  solar2025,
                        step_s=5*60,
                        selected_day=1,
                        start_hour=start_hour,
                        end_hour=end_hour,
                        acquistion_speed_fps=3,
                        processing_speed_fps=4,
                        seed=SEED,
                        state_content=state_content,
                        random_reset=False,
                        terminated_days=args.term_days,
                        forecast_steps=args.forecast_steps,
                        choose_forecast=args.choose_forecast,
                        latent_size=args.latent_size,
                        train_days=args.train_days,
                        start_threshold=args.start_thr,
                        prevision_noise_amount=args.prevision_noise,
                        battery_wh=battery_wh,
                        only_day_acquisition=args.only_day_acquisition,
                        battery_weight=args.battery_weight,
                        device_idle_energy_w=args.device_idle_energy_w,
                        device_full_energy_w=args.device_full_energy_w,
                        buffer_weight=args.buffer_weight,
                        random_day_switch=False,
                        discrete_action=args.discrete_action,
                        reward_shape=args.reward_shape,
                        buffer_incoming=args.buffer_incoming,
                        nn_model=nn_model_instance,
                        nn_lookback=args.nn_lookback,
                        nn_horizon=args.nn_horizon,
                        nn_min_val=nn_min_val,
                        nn_max_val=nn_max_val,
                        probabilistic_forecast=args.probabilistic_forecast,
                        nn_use_log1p=nn_use_log1p)

    if args.lr_decay is None:
        lr = args.lr
        print(f"Fixed LR: {lr}")
    elif args.lr_decay == "exp":
        lr = exp_schedule(args.lr)
        print("Exponential LR descrese")
    elif args.lr_decay == "cos":   
        lr = cosine_schedule(args.lr)
        print("Cosine LR descrese")
    elif args.lr_decay == "lin":   
        lr = linear_schedule(args.lr)
        print("Linear LR descrese")
    elif args.lr_decay == "lin_half":   
        lr = linear_schedule(args.lr, start_from=0.5)
        print("Linear LR from half descrese")
    elif args.lr_decay == "lin_thirth":   
        lr = linear_schedule(args.lr, start_from=0.3)
        print("Linear LR from thirth descrese")
        

    alg_entry = None
    if args.alg == "ppo":
        alg_entry = partial(PPO,
                            n_epochs=args.epochs,
                            n_steps = args.update_steps,
                            batch_size=256,
                            vf_coef=0.5,
                            clip_range=0.1,
                            #clip_range_vf=linear_schedule(0.5,0.3,end_value=0.1),
                            gamma=args.gamma,
    )
    elif args.alg == "a2c":
        alg_entry = partial(A2C,
                            n_steps = args.update_steps,
                            ent_coef = 0.01
    )
    elif args.alg == "dqn":
        alg_entry = partial(DQN,
                            train_freq=args.update_steps)
    elif args.alg == "rec_ppo":
        alg_entry = partial(RecurrentPPO,
                            n_epochs=args.epochs,
                            n_steps = args.update_steps,
                            batch_size=256,
                            vf_coef=0.5,
                            clip_range_vf=linear_schedule(0.5,0.3,end_value=0.1),
                            )
        
    policy = "MlpLstmPolicy" if args.alg == "rec_ppo" else "MlpPolicy"
    
    # We disable ForecastEmbedded for nn_prediction because PPO struggles to train a bottleneck 
    # layer from scratch when the features have some noise. Feeding them directly to the MLP works much better.
    use_forecast_extractor = args.use_embed_prev_day
    if use_forecast_extractor and (args.forecast_steps != args.latent_size):
        forecast_dim = args.forecast_steps
        extractor_kwargs = dict(
            normal_dim=test_env.observation_space.shape[0] - forecast_dim, 
            forecast_dim=forecast_dim, 
            latent_dim=args.latent_size
        )
        policy_kwargs = dict(
            net_arch=[args.layer_width] * args.layer_depth,
            features_extractor_class=ForecastEmbedded,
            features_extractor_kwargs=extractor_kwargs
        )
        print("Using ForecastEmbedded features extractor:", extractor_kwargs)
    else:
        policy_kwargs = dict(
            net_arch=[args.layer_width] * args.layer_depth,
        )
    print("Network", policy_kwargs)
    model = alg_entry(
        policy=policy,
        env = vec_env,
        device=device,
        learning_rate=lr,
        verbose=2,
        policy_kwargs=policy_kwargs,
        tensorboard_log=args.run_folder,
        seed=SEED
    )
    print("Model created")

    if not args.val:
        try:
            #reward_callback = TrainingRewardCallback(args.run_folder, verbose=1)
            model.learn(total_timesteps=args.steps, progress_bar=False)# callback=eval_callback)
            if args.run_folder is not None:
                model_path = os.path.join(args.run_folder,"last.pth")
                print("Model saved in",model_path)
                model.save(model_path)

            #log_path = os.path.join(args.run_folder, f"monitor_{0}.monitor.csv")
            #plot_results(log_path)
            print("Trainign completed")
        except Exception as e:
            print(e)
            if args.run_folder:
                print("Cleaning run folder",args.run_folder)
                shutil.rmtree(args.run_folder)

    if args.run_folder:
        path_to_load = None
        best_model_path = os.path.join(args.run_folder, 'best_model')
        last_model_path = os.path.join(args.run_folder, "last.pth")
        if args.val and args.model is not None:
            print("Loading model",args.model)
            path_to_load = args.model
        elif os.path.exists(best_model_path):
            print("Loading model from",best_model_path)
            path_to_load = best_model_path
        else:
            print("Loading model from",last_model_path)
            path_to_load = last_model_path

        if args.alg == "ppo":
            model = PPO.load(path_to_load)
        elif args.alg == "a2c":
            model = A2C.load(path_to_load)
        elif args.alg == "dqn":
            model = DQN.load(path_to_load)
        elif args.alg == "rec_ppo":
            model = RecurrentPPO.load(path_to_load)
        
    def print_day_img(fields:list[dict], datetimes:list, titles:list[str], path:str|list[str]):
        import matplotlib.dates as mdates
        plt.rcParams.update({'font.size': 14}) # Scegli un valore più grande, ad esempio 14 o 16
        plt.rcParams['axes.labelsize'] = 16 # Aumenta solo le etichette degli assi (Energy, Time Step)
        plt.rcParams['legend.fontsize'] = 14 # Aumenta solo la dimensione della legenda
        WIDTH = 2
        plt.rcParams['axes.linewidth'] = WIDTH 

        # Spessore delle linee dei tick (le piccole lineette sugli assi)
        plt.rcParams['xtick.major.width'] = WIDTH
        plt.rcParams['ytick.major.width'] = WIDTH

        fig, axs = plt.subplots(figsize=(14, 8), nrows=len(fields), squeeze=False)
        for i in range(len(fields)):
            ax = axs[i, 0]
            for f_name in fields[i].keys():
                #print("PRE",titles[i],"Field",f_name,"DT",len(datetimes[i]), "RL",len(fields[i][f_name]))
                if  len(datetimes[i]) > len(fields[i][f_name]):
                    dt = datetimes[i][:len(fields[i][f_name])]
                else:
                    dt = datetimes[i]
                #print("POST",titles[i],"Field",f_name,"DT",len(datetimes[i]), "RL",len(fields[i][f_name]))
                ax.step(dt, fields[i][f_name], label=f_name, linewidth=WIDTH, where="post")
            ax.legend(loc="upper left")
            ax.grid(True)
            ax.set_title(titles[i])
            
            # Format X-axis based on data type
            import datetime as dt_module
            if len(dt) > 0 and isinstance(dt[0], (dt_module.datetime, dt_module.date)):
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
                ax.set_xlim(left=dt[0], right=dt[-1])
            else:
                ax.set_xticks(ax.get_xticks()[::12])
                ax.set_xlim(left=0)
                
            ax.set_xlabel("Time")
            
        fig.autofmt_xdate()
        plt.tight_layout()
        #tikzplotlib.save("test.tex")
        if type(path)==list:
            for p in path:
                plt.savefig(p)
        else:
            plt.savefig(path)
        #plt.show()
        plt.close(fig)
    max_days = 7#len(solar2025.values)//(24*60//5)
    failed_days = 0
    processed_steps = 0
    inference_total_time_s = 0
    step_total_time_s = 0

    # Accumulate data across all days for a single continuous plot
    all_actions = []
    all_battery = []
    all_solar = []
    all_buffer = []
    all_datetimes = []

    # Override terminated_days so the env runs continuously for max_days
    test_env.terminated_days = max_days
    test_obs, info = test_env.reset(options={"norandom":True, "day":0})
    all_solar.append(test_obs[1])

    while processed_steps < (30*24*60//5):
        start_time = time()
        action,_ = model.predict(test_obs, deterministic=True)
        inference_total_time_s += (time()-start_time)
        start_time = time()
        test_obs, reward, done, terminated, info = test_env.step(action)
        step_total_time_s += (time() - start_time)
        processed_steps += 1

        # Collect data
        for value, name in list(zip(test_obs, info["fields"])):
            if name == "Battery":
                all_battery.append(value.item())
            elif name == "Solar":
                all_solar.append(value.item())
            elif name == "Memory":
                all_buffer.append(value.item())
        if "cons" in info:
            all_actions.append(info["cons"])
        all_datetimes.append(info["time"])

        if done and terminated==False:
            failed_days += 1
        if done:
            break

    # Generate single continuous plot
    if args.run_folder:
        filename = os.path.join(args.run_folder,"test_images")
        os.makedirs(filename, exist_ok=True)
        filename_png = os.path.join(filename, "continuous.png")
        filename_pdf = os.path.join(filename, "continuous.pdf")

        min_len = min(len(all_actions), len(all_battery), len(all_solar), len(all_buffer))
        day_fields = []
        day_fields.append({
            "Action": all_actions[:min_len],
            "Battery": all_battery[:min_len],
            "Solar": all_solar[:min_len],
            "Buffer": all_buffer[:min_len],
        })
        print_day_img(day_fields, [all_datetimes[:min_len]], ["RL"], [filename_png, filename_pdf])
        print(f"Plot saved to {filename_png}")

    print("Early shutdown", failed_days)
    print("Processed steps",processed_steps)
    print("Processed images", test_env.processed_images)
    print("AVG inference time (ms)",1000*inference_total_time_s/processed_steps)
    print("AVG step time (ms)",1000*step_total_time_s/processed_steps)

    
    print("Measuring flops")
    policy_module = model.policy.cpu()
    obs_shape = test_env.observation_space.shape
    dummy_input = torch.zeros(1, *obs_shape, dtype=torch.float32)
    inputs = (dummy_input,)
    policy_flops, policy_params = profile(policy_module, inputs=inputs, verbose=False)
    policy_flops_str, policy_params_str = clever_format([policy_flops, policy_params], "%.3f")
    print(f"FLOPs (forward pass): {policy_flops_str}")
    #print(f"Parameters: {policy_params_str}")

    if test_env.linear_mlp is not None:
        policy_module = test_env.linear_mlp.cpu()
        print(policy_module)
        dummy_input = torch.zeros(1, 128, dtype=torch.float32)
        inputs = (dummy_input,)
        embed_flops, embed_params = profile(policy_module, inputs=inputs, verbose=False)
        str_embed_flops, embed_params = clever_format([embed_flops, embed_params], "%.3f")
        print(f"Embedding FLOPs",str_embed_flops)
        total_flops = clever_format([policy_flops+embed_flops], "%.3f")
        print(f"Total FLOPs",total_flops)

    exit(0)

if __name__ == "__main__":
    main()
"""
processed_images = []
test_obs, info = test_env.reset(options={"norandom":True})
battery_levels, recharges, consumptions, times, forecast = [], [], [], [], []
battery_levels.append(test_obs[0])
done = False
rewards = []
is_plotted = False
plot_time = 60*60*24*31
unprocessed_images = 0
datetimes = []
fields = {}
for f in info["fields"]:
    fields[f]=[]
fields["Energy"]=[]

if test_env.choose_forecast:
    fields["start_time"]=[]
    fields["window_time_s"]=[]

# Remove fields
needed_views = ["Battery","Solar","Hour", "Energy","Hour Minute","Buffer","Sunset"]
print("Start time",test_env.time_s)
while True:
    # if processing_buffer:
    #     test_obs, reward, done, terminated, info = test_env.clear_buffer()
    #     if terminated or done:
    #         processing_buffer = False
    # else:
    action,_ = model.predict(test_obs, deterministic=True)
    test_obs, reward, done, terminated, info = test_env.step(action)
    #processing_buffer = terminated and test_env.buffer_length > 0
    rewards.append(reward)
    datetimes.append(info["time"])

    for value,name in list(zip(test_obs, fields.keys())):
        fields[name].append(value)
    if "Energy" in fields.keys(): fields["Energy"].append(info["cons"])
    if test_env.choose_forecast:
        fields["start_time"].append(info["forecast"][0])
        fields["window_time_s"].append(info["forecast"][1])

    #if steps % 1000 == 0: print("Processing day",test_env.get_uptime_s()//(60*60*24),"of 121 | Processed images",test_env.processed_images, end="\r")
    if args.run_folder != "../runs" and (test_env.get_uptime_s() >= plot_time or done):
        plot_time += 60*60*24*31
        month = int(test_env.get_uptime_s() // (60*60*24*31))
        print("Plotting", month)
        # Limit da for 1 week
        steps_for_1_week = (60*60*24*7)//300
        for f in fields.keys():
            fields[f]=fields[f][:steps_for_1_week]
        image_folder = os.path.join(args.run_folder,"images")
        os.makedirs(image_folder, exist_ok=True)
        img_file = os.path.join(image_folder,f"image{month}.jpg")
        datetimes = list(map(lambda x:x.strftime("%H:%M"), datetimes))
        save_plot(fields, img_file, needed_views, x_values=datetimes)
        for f in fields.keys():
            fields[f].clear()
    if done or test_env.get_uptime_s() >= 120*24*60*60:
        print()
        print("BATTERY DEPLETED!",done,terminated)
        break
    if terminated:
        image_folder = os.path.join(args.run_folder,"images")
        os.makedirs(image_folder, exist_ok=True)
        img_file = os.path.join(image_folder,f"image.jpg")
        datetimes = list(map(lambda x:x.strftime("%H:%M"), datetimes))
        save_plot(fields, img_file, needed_views, x_values=datetimes)
        for f in fields.keys():
            fields[f].clear()
        break

print("End time",test_env.time_s)
print("Time reached", test_env.get_human_uptime(), test_env.get_uptime_s())
#print("Final reward", sum(rewards),len(rewards))
print("Processed images", test_env.processed_images)
print("Buffer images",test_env.buffer_length)
processed_images.append(test_env.processed_images)

print(processed_images)
"""