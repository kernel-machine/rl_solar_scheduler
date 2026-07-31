import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from lib.solar.prediction import GHIPredictorLSTM, GHIPredictorTCN, GHIPredictorTransformer

def create_multivariate_sequences(features, target_col_idx, lookback, horizon):
    num_samples = len(features) - lookback - horizon + 1
    shape_X = (num_samples, lookback, features.shape[1])
    strides_X = (features.strides[0], features.strides[0], features.strides[1])
    X = np.lib.stride_tricks.as_strided(features, shape=shape_X, strides=strides_X)

    target = features[:, target_col_idx]
    shape_y = (num_samples, horizon)
    strides_y = (target.strides[0], target.strides[0])
    y = np.lib.stride_tricks.as_strided(target[lookback:], shape=shape_y, strides=strides_y)

    return X.copy(), y.copy()

def quantile_loss(preds, target, quantiles=[0.1, 0.5, 0.9]):
    # preds: (batch, horizon * len(quantiles))
    # target: (batch, horizon)
    horizon = target.size(1)
    preds = preds.view(-1, horizon, len(quantiles))
    target = target.unsqueeze(-1)
    
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, :, i:i+1]
        loss_q = torch.max((q - 1) * errors, q * errors)
        losses.append(loss_q)
        
    return torch.mean(torch.sum(torch.cat(losses, dim=-1), dim=1))

def main():
    parser = argparse.ArgumentParser(description="Predictor for GHI/GTI prediction")
    parser.add_argument("dataset", type=str, help="Path to the dataset CSV file")
    parser.add_argument("--model-type", type=str, default="lstm", choices=["lstm", "tcn", "transformer"], help="Model architecture")
    parser.add_argument("--probabilistic", action="store_true", help="Enable quantile probabilistic forecasting (P10, P50, P90)")
    parser.add_argument("--lookback", type=int, default=96, help="Lookback window size (e.g. 96 for 8h at 5min intervals)")
    parser.add_argument("--horizon", type=int, default=24, help="Prediction horizon (e.g. 24 for 2h)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden size")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--use-log1p", action="store_true", help="Apply log1p transformation to GHI/GTI values")
    parser.add_argument("--output-model", type=str, default="ghi_predictor.pth", help="Output model path")
    parser.add_argument("--output-plot", type=str, default="ghi_prediction_comparison.png", help="Output plot path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    df = pd.read_csv(args.dataset)
    if 'gti' in df.columns:
        col_name = 'gti'
    elif 'ghi' in df.columns:
        col_name = 'ghi'
    else:
        raise ValueError("CSV must contain a 'gti' or 'ghi' column")

    gti_values = df[col_name].values.astype(np.float32)

    if args.use_log1p:
        gti_values = np.log1p(gti_values)

    min_val, max_val = float(np.min(gti_values)), float(np.max(gti_values))
    if max_val > min_val:
        gti_scaled = (gti_values - min_val) / (max_val - min_val)
    else:
        gti_scaled = gti_values

    if 'period_end' in df.columns:
        time_str = df['period_end'].str[11:16]
        hours = time_str.str[:2].astype(float) + time_str.str[3:5].astype(float) / 60.0
        hours = hours.values
    else:
        hours = (np.arange(len(df)) % 288) * (5.0 / 60.0)

    sin_hour = np.sin(2 * np.pi * hours / 24.0).astype(np.float32)
    cos_hour = np.cos(2 * np.pi * hours / 24.0).astype(np.float32)

    feature_matrix = np.column_stack([gti_scaled, sin_hour, cos_hour]).astype(np.float32)
    num_features = feature_matrix.shape[1]

    X, y = create_multivariate_sequences(feature_matrix, target_col_idx=0, lookback=args.lookback, horizon=args.horizon)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)

    actual_output_size = args.horizon * 3 if args.probabilistic else args.horizon

    if args.model_type == "lstm":
        model = GHIPredictorLSTM(input_size=num_features, hidden_size=args.hidden_size, num_layers=args.num_layers, output_size=actual_output_size).to(device)
    elif args.model_type == "tcn":
        model = GHIPredictorTCN(input_size=num_features, hidden_size=args.hidden_size, num_layers=args.num_layers, output_size=actual_output_size).to(device)
    elif args.model_type == "transformer":
        model = GHIPredictorTransformer(input_size=num_features, hidden_size=args.hidden_size, num_layers=args.num_layers, output_size=actual_output_size).to(device)

    criterion = quantile_loss if args.probabilistic else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    n_train = X_train_t.size(0)
    batch_size = args.batch_size

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        train_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            batch_X, batch_y = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {train_loss:.6f}")

    model_name = args.output_model + '_' + args.model_type + '.pth'
    torch.save(model.state_dict(), model_name)
    print(f"Model saved to '{model_name}'")

    scaling_path = model_name.replace('.pth', '_scaling.npz')
    np.savez(scaling_path, min_val=min_val, max_val=max_val,
             use_log1p=args.use_log1p, lookback=args.lookback, horizon=args.horizon,
             hidden_size=args.hidden_size, num_layers=args.num_layers, input_size=num_features,
             model_type=args.model_type, probabilistic=args.probabilistic)
    print(f"Scaling parameters saved to '{scaling_path}'")

    # Evaluation loss on test set
    print("Starting Evaluation on Test Set...")
    model.eval()
    test_loss = 0.0
    n_test = X_test_t.size(0)
    n_test_batches = 0

    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            batch_X, batch_y = X_test_t[i:i + batch_size], y_test_t[i:i + batch_size]
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            n_test_batches += 1

    test_loss /= n_test_batches
    print(f"Test Loss: {test_loss:.6f}")

    # Full-Day Rolling Evaluation for 4 Full Days in Test Set
    print("Generating Full-Day Evaluation Plots...")
    steps_per_day = 288  # 24 hours at 5-minute steps
    test_start_idx = split + args.lookback
    total_test_rows = len(df) - test_start_idx
    num_test_days = total_test_rows // steps_per_day

    # Pick 4 distinct full days spread across the test set
    day_indices = [0, max(1, num_test_days // 4), max(2, num_test_days // 2), max(3, (3 * num_test_days) // 4)]

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    axs = axs.flatten()
    time_hours = np.linspace(0, 24, steps_per_day, endpoint=False)

    for i, day_idx in enumerate(day_indices):
        day_start_row = test_start_idx + day_idx * steps_per_day
        day_end_row = day_start_row + steps_per_day

        if day_end_row > len(df):
            break

        real_day_gti = df[col_name].values[day_start_row:day_end_row]

        # Batch rolling evaluation for the entire 24h day
        day_windows = []
        for step_t in range(day_start_row, day_end_row):
            win = feature_matrix[step_t - args.lookback:step_t]
            day_windows.append(win)

        day_windows_t = torch.from_numpy(np.array(day_windows, dtype=np.float32)).to(device)

        with torch.no_grad():
            preds_scaled = model(day_windows_t).cpu().numpy()
            # 1-step ahead rolling prediction for every 5-min step in the day
            if args.probabilistic:
                pred_1step_scaled = preds_scaled[:, 1] # P50
            else:
                pred_1step_scaled = preds_scaled[:, 0]

        pred_day_gti = pred_1step_scaled * (max_val - min_val) + min_val
        if args.use_log1p:
            pred_day_gti = np.expm1(pred_day_gti)

        if 'period_end' in df.columns:
            date_str = str(df['period_end'].iloc[day_start_row])[:10]
        else:
            date_str = f"Day {day_idx}"

        ax = axs[i]
        ax.plot(time_hours, real_day_gti, label="Real GTI", color='tab:blue', linewidth=2)
        ax.plot(time_hours, pred_day_gti, label="Prediction (24h)", color='tab:orange', linestyle='--', linewidth=2)
        
        if args.probabilistic:
            p10_scaled = preds_scaled[:, 0]
            p90_scaled = preds_scaled[:, 2]
            p10 = p10_scaled * (max_val - min_val) + min_val
            p90 = p90_scaled * (max_val - min_val) + min_val
            if args.use_log1p:
                p10 = np.expm1(p10)
                p90 = np.expm1(p90)
            ax.fill_between(time_hours, p10, p90, color='tab:orange', alpha=0.3, label="P10 - P90")

        ax.set_title(f"Full-Day Solar Prediction (24h) - Date: {date_str}")
        ax.set_xlabel("Hour of Day (00:00 - 24:00)")
        ax.set_ylabel("GTI Value (W/m²)")
        ax.set_xlim(0, 24)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plot_name = args.output_plot.replace('.png', f'_{args.model_type}.png')
    plt.savefig(plot_name)
    print(f"Full-day comparison plot saved in '{plot_name}'")

if __name__ == "__main__":
    main()
