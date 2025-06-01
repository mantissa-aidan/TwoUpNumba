import numpy as np
from stable_baselines3 import PPO
from two_up_env_numba import TwoUpEnvNumba
import argparse
import os

def main(total_timesteps=100000, personality='standard'):
    """Train an agent with a specific personality and save it."""
    
    # Define base paths dynamically based on personality
    base_output_path = f"personality_agents/{personality}"
    model_dir = f"{base_output_path}/models"
    tensorboard_log_dir = f"{base_output_path}/tensorboard_logs"

    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    print(f"Initializing environment with personality: {personality}")
    train_env = TwoUpEnvNumba(gambler_personality=personality)

    print(f"Training PPO agent ({personality} personality) for {total_timesteps} timesteps...")
    model = PPO("MlpPolicy", train_env, verbose=1, device="cpu", tensorboard_log=tensorboard_log_dir)
    model.learn(total_timesteps=total_timesteps, tb_log_name=f"ppo_two_up_{personality}", progress_bar=True)
    print(f"Training complete.")
    
    model_save_path = f"{model_dir}/ppo_two_up_{personality}"
        
    model.save(model_save_path)
    print(f"Trained model saved to {model_save_path}.zip")
    print(f"TensorBoard logs saved in: {tensorboard_log_dir}/ppo_two_up_{personality}_1")
    print(f"\nTo evaluate this model, you can use a separate evaluation script.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a PPO agent with a specific personality for TwoUp.')
    parser.add_argument('--timesteps', type=int, default=100000, help='Number of timesteps to train the PPO agent.')
    parser.add_argument('--personality', type=str, default='standard', choices=['standard', 'thrill_seeker', 'loss_averse', 'play_for_time'], help='Gambler personality to simulate during training.')
    args = parser.parse_args()

    main(total_timesteps=args.timesteps, personality=args.personality) 