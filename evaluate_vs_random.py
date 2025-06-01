import numpy as np
from stable_baselines3 import PPO
from two_up_env_numba import TwoUpEnvNumba
from collections import defaultdict
import argparse
import os
import json
import datetime

def run_simulation(env, model=None, num_episodes=1000, is_random_agent=False):
    """Run multiple episodes and collect statistics for a given agent."""
    results = defaultdict(list)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            if is_random_agent:
                action = env.action_space.sample()  # 0: no bet, 1: heads, 2: tails
            elif model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else: # Fallback, though should ideally not be reached if model is specified
                action = env.action_space.sample() 

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            if done:
                results['final_balance'].append(info.get('balance', env.balance))
                results['win_rate'].append(info.get('win_rate', 0.0))
                results['episode_reward'].append(episode_reward)
                results['total_games'].append(info.get('total_games', env.total_games))
    
    return results

def main(model_path=None, personality='standard', num_episodes_test=1000):
    """Load a trained PPO agent and compare it against a random agent."""
    
    # Infer model_path if not provided
    if model_path is None:
        model_path = f"personality_agents/{personality}/models/ppo_two_up_{personality}.zip"
        print(f"Model path not provided, inferring based on personality: {model_path}")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading PPO model from: {model_path}")
    try:
        model = PPO.load(model_path, device="cpu") # Load to CPU for consistency
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Initializing environment with personality: {personality}")
    # Environment for testing the PPO model
    test_env_ppo = TwoUpEnvNumba(gambler_personality=personality)
    # Environment for testing the random agent (using same personality for consistent env dynamics)
    test_env_random = TwoUpEnvNumba(gambler_personality=personality)

    # Run simulations
    print(f"\nRunning simulations for {num_episodes_test} episodes each (environment personality: {personality})...")
    
    print(f"Evaluating trained PPO agent ({personality} personality)...")
    ppo_results = run_simulation(test_env_ppo, model=model, num_episodes=num_episodes_test)
    
    print(f"Evaluating random agent (in {personality} context environment)..." )
    random_results = run_simulation(test_env_random, num_episodes=num_episodes_test, is_random_agent=True)
    
    # Print statistics
    print("\n--- Results Summary ---")
    print(f"Personality Profile for Environment: {personality}")
    print(f"Model Path: {model_path}")
    
    print("\nTrained PPO Agent:")
    print(f"  Average Final Balance: ${np.mean(ppo_results['final_balance']):.2f}")
    print(f"  Average Win Rate: {np.mean(ppo_results['win_rate']):.2%}")
    print(f"  Average Episode Reward: {np.mean(ppo_results['episode_reward']):.2f}")
    print(f"  Average Games Played per Episode: {np.mean(ppo_results['total_games']):.2f}")

    print("\nRandom Agent:")
    print(f"  Average Final Balance: ${np.mean(random_results['final_balance']):.2f}")
    print(f"  Average Win Rate: {np.mean(random_results['win_rate']):.2%}")
    print(f"  Average Episode Reward: {np.mean(random_results['episode_reward']):.2f}")
    print(f"  Average Games Played per Episode: {np.mean(random_results['total_games']):.2f}")

    # Prepare data for saving
    output_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_path": model_path,
        "personality_profile": personality,
        "num_episodes_test": num_episodes_test,
        "ppo_agent_results": {
            "average_final_balance": np.mean(ppo_results['final_balance']),
            "average_win_rate": np.mean(ppo_results['win_rate']),
            "average_episode_reward": np.mean(ppo_results['episode_reward']),
            "average_games_played_per_episode": np.mean(ppo_results['total_games'])
        },
        "random_agent_results": {
            "average_final_balance": np.mean(random_results['final_balance']),
            "average_win_rate": np.mean(random_results['win_rate']),
            "average_episode_reward": np.mean(random_results['episode_reward']),
            "average_games_played_per_episode": np.mean(random_results['total_games'])
        }
    }

    # Save results to JSON file
    report_dir = f"personality_agents/{personality}/reports"
    os.makedirs(report_dir, exist_ok=True)
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{report_dir}/evaluation_summary_{timestamp_str}.json"
    
    with open(report_filename, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"\nResults also saved to: {report_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained PPO agent against a random agent for TwoUp.')
    parser.add_argument('--model_path', type=str, required=False, default=None, help='Path to the trained PPO model (.zip file). If not provided, path is inferred from personality.')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to run for testing each agent.')
    parser.add_argument('--personality', type=str, default='standard', choices=['standard', 'thrill_seeker', 'loss_averse', 'play_for_time'], help='Gambler personality context for the environment.')
    args = parser.parse_args()

    main(model_path=args.model_path, personality=args.personality, num_episodes_test=args.episodes) 