from src.config import Config
from src.nuscenes_loader import process_nuscenes_subset
from src.train.train_lstm import train_lstm
from src.train.train_fusion import train_fusion

def main():
    config = Config()
    config.nuscenes_version = "v1.0-mini"
    config.num_epochs = 30

    print("Loading data...")
    bevs, imu, gps, targets = process_nuscenes_subset(config)

    print("\n--- Training LSTM baseline ---")
    lstm_pred = train_lstm(config, imu, targets)

    print("\n--- Training Fusion Model ---")
    fusion_model, fusion_pred = train_fusion(config, bevs, imu, targets)

if __name__ == "__main__":
    main()