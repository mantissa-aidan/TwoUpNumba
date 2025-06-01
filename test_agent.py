from stable_baselines3 import PPO
from two_up_env import TwoUpEnv
import torch
import argparse

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
    parser = argparse.ArgumentParser(description='Test a trained Two Up agent')
    parser.add_argument('--model_path', type=str, default='trained_models/ppo_two_up',
                      help='Path to the trained model')
    parser.add_argument('--episodes', type=int, default=10,
                      help='Number of episodes to run')
    args = parser.parse_args()

    # Set up device
    device = get_device()
    print(f"Using device: {device}")

    # Create the environment
    env = TwoUpEnv(render_mode='human')

    # Load the trained model
    model = PPO.load(args.model_path, device=device)
    print(f"Loaded model from {args.model_path}")

    # Run episodes
    for episode in range(args.episodes):
        print(f"\nEpisode {episode + 1}/{args.episodes}")
        obs, info = env.reset()
        total_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env.render()

        print(f"Episode {episode + 1} - Final balance: ${info['balance']}")
        print(f"Episode {episode + 1} - Total reward: {total_reward}")
        print(f"Episode {episode + 1} - Win rate: {info['win_rate']:.2%}")

if __name__ == "__main__":
    main() 