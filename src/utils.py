import math
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gymnasium as gym
import os
import numpy as np
import random
from scipy.ndimage import gaussian_filter1d
import matplotlib as mpl
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common import results_plotter

class ForecastEmbedded(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, normal_dim:int, forecast_dim:int, latent_dim: int):
        super().__init__(observation_space, features_dim=normal_dim+latent_dim)

        assert observation_space.shape[0] == normal_dim + forecast_dim, (
            f"Observation space ({observation_space.shape[0]}) "
            f"is different than ({normal_dim}) + raw_dim ({forecast_dim})."
        )

        self.normal_dim = normal_dim
        self.forecast_dim = forecast_dim
        
        hidden_dim = 64
        self.latent_encoder = torch.nn.Sequential(
            torch.nn.Linear(forecast_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, latent_dim)
        )

        for module in self.latent_encoder.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.orthogonal_(module.weight, gain=1.414)
                torch.nn.init.constant_(module.bias, 0.0)


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        normal_features = observations[:,:self.normal_dim] # Takes normal features
        raw_features = observations[:, self.normal_dim:] # Takes raw features
        latent_features = self.latent_encoder(raw_features)
        combined_features = torch.cat((normal_features, latent_features), dim=1)
        return combined_features
    
def linear_schedule(initial_value: float, start_from: float = 0.0, end_value: float = 0.0):
    def func(progress_remaining: float) -> float:
        elapsed_progress = 1.0 - progress_remaining

        if elapsed_progress < start_from:
            return initial_value
        else:
            progress_in_decay_window = elapsed_progress - start_from
            decay_window_duration = 1.0 - start_from
            
            if decay_window_duration == 0:
                fraction_scaling = 1.0 # Avoid 0 division
            else:
                fraction_scaling = progress_in_decay_window / decay_window_duration
            
            valore_scalato = initial_value + fraction_scaling * (end_value - initial_value)
            
            return max(valore_scalato, end_value) 
            
    return func

def exp_schedule(initial_value, decay_rate=5):
    def func(progress_remaining: float) -> float:
        return initial_value * math.exp(-decay_rate * (1 - progress_remaining))
    return func

def cosine_schedule(initial_value):
    def func(progress_remaining: float) -> float:
        return initial_value * 0.5 * (1 + math.cos(math.pi * (1 - progress_remaining)))
    return func

def save_plot(fields:dict, path:str|list, views:list = None, x_values:list[datetime] = None,
               custom_alpha:dict = None, custom_linestyle:dict = None, custom_lambda:callable=None, y_label:str=None, x_label:str="Time Step"):

    plt.rcParams.update({'font.size': 14})
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 14 
    WIDTH = 2
    plt.rcParams['axes.linewidth'] = WIDTH 

    plt.rcParams['xtick.major.width'] = WIDTH
    plt.rcParams['ytick.major.width'] = WIDTH
    
    fig, ax = plt.subplots(figsize=(12, 5.5)) 
    for f_name in fields.keys():
        alpha = 1
        linestyle = "solid"
        if custom_alpha is not None and f_name in custom_alpha.keys():
            alpha = custom_alpha[f_name]
        if custom_linestyle is not None and f_name in custom_linestyle.keys():
            linestyle = custom_linestyle[f_name]
        if views is None:
            if x_values is not None:
                drawstyle = None
                if f_name == "Energy":
                    drawstyle = 'steps-post'
                ax.step(x_values, fields[f_name], label=f_name, drawstyle=drawstyle, alpha=alpha, linestyle=linestyle, linewidth=WIDTH, where="post")
            else:
                ax.step(range(len(fields[f_name])), fields[f_name], label=f_name, alpha=alpha, linestyle=linestyle, linewidth=WIDTH, where="post")
        elif views is not None and f_name in views:
            if x_values is not None:
                ax.step(x_values, fields[f_name], label=f_name, alpha=alpha, linestyle=linestyle, linewidth=WIDTH, where="post")
            else:
                ax.step(range(len(fields[f_name])), fields[f_name], label=f_name, alpha=alpha, linestyle=linestyle, linewidth=WIDTH, where="post")
    if custom_lambda is not None:
        custom_lambda(ax)
    ax.set_xlim(left=0)
    ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)

    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique.keys():
            unique[label] = handle
    unique_handles = unique.values()
    unique_labels = unique.keys()
    plt.legend(unique_handles, unique_labels)

    plt.grid(True)
    plt.tight_layout()
    if type(path)==list:
        for e in path:
            plt.savefig(e)
    else:
        plt.savefig(path)
    plt.close(fig)

class MyEvalCallBack(BaseCallback):
    def __init__(self, env : gym.Env, best_model_path:str, eval_freq:int = 0, verbose = 0):
        super().__init__(verbose)
        self.env = env
        self.best_images = 0
        self.best_model_path = best_model_path
        self.eval_freq = eval_freq
        self.next_call = eval_freq

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls > self.next_call:
            print("Triggered callback",self.n_calls, self.next_call)
            self.next_call+=self.eval_freq
            obs, _ = self.env.reset()
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                if done:# or truncated:
                    if self.env.processed_images > self.best_images:
                        self.best_images = self.env.processed_images
                        self.model.save(self.best_model_path)
                        self.logger.record("env_images",self.env.processed_images)
                        print(f"New best model with {self.env.processed_images} processed images")
                    break
        return True

def signal_noise(signal:list[float], strength:float=0.2, max_value:float=1, rng:random.Random = None) -> list[float]:
    if strength == 0:
        return signal
    
    if rng is None:
        rng = random.Random()
    
    def apply_noise(x):
        if x > 0:
            return x+strength*2*(rng.random()-0.5)
        else:
            return 0
    signal_noised = list(map(apply_noise, signal))
    signal_smossed = gaussian_filter1d(signal_noised, sigma=2)
    max_value = max(signal_smossed)
    if max_value>0:
        scale = max_value/max_value
    else:
        scale = 1
    signal_smossed = list(map(lambda x:max(0,min(x*scale,1)), signal_smossed))
    signal_smossed = np.asarray(signal_smossed)
    non_zero_indices = np.where(signal_smossed > 0.05)[0]
    if non_zero_indices.size > 0:
        first_significant_idx = non_zero_indices[0]
        last_significant_idx = non_zero_indices[-1]
        signal_smossed[:first_significant_idx] = 0.0
        signal_smossed[last_significant_idx + 1:] = 0.0
    return signal_smossed

def plot_results(log_folder, title='Learning Curve'):
    x, y = results_plotter.ts2xy(log_folder, 'timesteps')
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.xlabel('Timesteps')
    plt.ylabel('Average Episode Rewards')
    plt.title(title)
    plt.grid()
    
    plot_file_path = f"{log_folder}/reward_plot.png"
    plt.savefig(plot_file_path)
