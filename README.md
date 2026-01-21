# Reinforcement Learning Scheduler for Solar-Powered Edge Devices
## Authors :pencil2:
- Luca Giovannesi
- Paolo Russo
- Roberto Beraldi
## Abstract :page_facing_up:
Edge devices powered by renewable energy sources,
such as solar panels, face the challenge of operating under uncertain and intermittent energy availability. This paper
propose a Reinforcement Learning (RL)  scheduler for heavy tasks, e.g. image classification running on solar-powered devices with an accumulator. 

The scheduler maximizes the total number of processed images by deciding whether to process an image immediately or store it for later processing, all while remaining energy-aware to prevent device shutdown. Crucially, the method operates without any prior knowledge of future solar energy production, making it suitable for devices lacking an internet connection for forecasting. The RL method processes 99.8% of the images processed by the optimal solution obtained through Integer Linear Programming (ILP), missing on average only 224 seconds of captured frames per day. This demonstrates that RL is a viable alternative to optimal solvers even in forecast-free scenarios.
## Setup the system :gear:

- Put [solcast2024.csv](https://github.com/user-attachments/files/21213151/solcast2024.csv) and [solcast2025.csv](https://github.com/user-attachments/files/21213140/solcast2025.csv) on the project folder (not in SRC) but in the project folder
- install the python libraries from the `requirements.txt`

## How to replicate the experiments :eyes:

Go to the `src` directory and run the `main.py`, the script accepts different parameters.

To reproduce the results of the paper you can run the software with the following parameters

For the PPO alghoritms:

```bash
python main.py --gpu --use_solar --use_hour_minute --use_images --steps 2000000 --update_steps 2048 --alg ppo --lr 0.0003 --term_days 1 --test_year 2025 --n_env 10 --layer_width 64 --lr_decay lin --train_days 30 --autostart --start_thr 0.05 --use_pressure --use_humidity
```

For the A2C alghoritms:

```bash
python -u main.py --gpu --use_solar --use_hour_minute --use_images --steps 2000000 --update_steps 30 --alg a2c --lr 0.0007 --term_days 1 --test_year 2025 --n_env 10 --layer_width 64 --lr_decay lin --train_days 30 --autostart --start_thr 0.05 --use_pressure --use_humidity
```

It is possible to include or not include the parameters `--use_pressure`,`--use_humidity` to tests all the different scenarios reported in the paper.

For each run is created a folder in `runs`, containing all the plots of the executed experiments.

## Important files :file_folder:
- **main.py** contains the training and evaluation code
- **ilp_solver.py** contains the code to find the optimal solutions using the ILP
- **lib/environment.py** contains the RL Environment
