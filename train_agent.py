from stable_baselines3 import PPO
from two_up_env import TwoUpEnv
import numpy as np
import torch
import os
from stable_baselines3.common.vec_env import SubprocVecEnv

def get_device():
    """
    Get the best available device for training.
    Returns:
        torch.device: The device to use for training
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # For Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")

def main():
    # Set up device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create the vectorized environment for training
    num_envs = 4  # You can increase this for more parallelism
    env = SubprocVecEnv([lambda: TwoUpEnv(render_mode=None) for _ in range(num_envs)])
    
    # Create the agent (PPO algorithm) with GPU support
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        tensorboard_log="./tensorboard_logs/"
    )
    
    # Train the agent
    print("Training the agent...")
    model.learn(
        total_timesteps=100000,
        progress_bar=True,
        tb_log_name="ppo_two_up"
    )
    
    # Save the trained model
    model_path = "trained_models/ppo_two_up"
    os.makedirs("trained_models", exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Test the trained agent using a single environment for rendering
    print("\nTesting the trained agent...")
    test_env = TwoUpEnv(render_mode='human')
    obs, info = test_env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        test_env.render()
    
    print(f"\nFinal balance: ${info['balance']}")
    print(f"Total reward: {total_reward}")

if __name__ == "__main__":
    main() 