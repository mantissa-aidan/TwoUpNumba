import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback
from two_up_env_numba import TwoUpEnvNumba
import argparse
import os

def main(total_timesteps=100000, personality='standard', 
         initial_balance=100.0, base_bet_amount=10.0, max_episode_steps=200,
         continue_training=False):
    """Train an agent with a specific personality and save it."""
    
    base_output_path = f"personality_agents/{personality}"
    model_dir = os.path.join(base_output_path, "models")
    tensorboard_log_dir = os.path.join(base_output_path, "tensorboard_logs")
    model_save_path_zip = os.path.join(model_dir, f"ppo_two_up_{personality}.zip")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    print(f"Initializing environment with personality: {personality}")
    print(f"Env params: Initial Balance: {initial_balance}, Base Bet Unit: {base_bet_amount}, Max Episode Steps: {max_episode_steps}")
    train_env = TwoUpEnvNumba(
        gambler_personality=personality,
        initial_balance=initial_balance,
        base_bet_amount=base_bet_amount,
        max_episode_steps=max_episode_steps
    )

    model_loaded = False
    if continue_training and os.path.exists(model_save_path_zip):
        print(f"Attempting to load existing model from: {model_save_path_zip}")
        try:
            model = PPO.load(model_save_path_zip, env=train_env, device="cpu", tensorboard_log=tensorboard_log_dir)
            model.set_env(train_env, force_reset=False) 
            print(f"Successfully loaded model. Continuing training for {total_timesteps} additional timesteps.")
            model_loaded = True
        except Exception as e:
            print(f"Error loading existing model: {e}. Creating a new model instead.")
            model_loaded = False
    
    if not model_loaded:
        if continue_training and not os.path.exists(model_save_path_zip):
            print(f"Continue training specified, but model not found at {model_save_path_zip}. Creating a new model.")
        elif not continue_training:
            print("Starting new training session.")

        model = PPO(
            "MlpPolicy", 
            train_env, 
            verbose=1, 
            device="cpu", 
            tensorboard_log=tensorboard_log_dir,
            learning_rate=0.0003
        )
        print(f"Training new PPO agent ({personality} personality) for {total_timesteps} timesteps...")

    progress_callback = ProgressBarCallback()
    model.learn(
        total_timesteps=total_timesteps, 
        tb_log_name=f"ppo_two_up_{personality}", 
        callback=progress_callback,
        reset_num_timesteps=not model_loaded
    )
    print(f"Training complete.")
    
    model.save(model_save_path_zip)
    print(f"Trained model saved to {model_save_path_zip}")
    print(f"TensorBoard logs will be in: {tensorboard_log_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a PPO agent with a specific personality for TwoUp.')
    parser.add_argument('--timesteps', type=int, default=100000, help='Number of timesteps to train (or continue training) the PPO agent.')
    parser.add_argument('--personality', type=str, default='standard', 
                        choices=['standard', 'thrill_seeker', 'loss_averse', 'play_for_time'], 
                        help='Gambler personality to simulate during training.')
    parser.add_argument("--initial-balance", type=float, default=100.0, help="Initial balance for the agent.")
    parser.add_argument("--base-bet", type=float, default=10.0, help="Base bet unit for the agent environment.")
    parser.add_argument("--max-episode-steps", type=int, default=200, help="Maximum steps per episode.")
    parser.add_argument("--continue-training", action='store_true', help="Load existing model and continue training.")

    args = parser.parse_args()

    main(
        total_timesteps=args.timesteps, 
        personality=args.personality,
        initial_balance=args.initial_balance,
        base_bet_amount=args.base_bet,
        max_episode_steps=args.max_episode_steps,
        continue_training=args.continue_training
    ) 