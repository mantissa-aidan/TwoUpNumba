from stable_baselines3 import PPO
from two_up_env_numba import TwoUpEnvNumba
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
import os
import datetime
import json
from tqdm import tqdm

def run_simulation(env, model=None, num_episodes=1000, always_no_bet=False, strategy_name="Agent"):
    """Run multiple episodes and collect statistics"""
    results = defaultdict(list)
    
    for episode in tqdm(range(num_episodes), desc=f"Simulating {strategy_name}"):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0 
        
        while not done:
            step_count += 1 
            if always_no_bet:
                action = 0
            elif model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            if done:
                results['final_balance'].append(info.get('balance', env.balance))
                results['win_rate'].append(info.get('win_rate', 0.0))
                results['episode_reward'].append(episode_reward)
                results['total_games'].append(info.get('total_games', env.total_games))
    
    return results

# Modified plot_results to handle PPO vs Baselines
def plot_results(ppo_results: defaultdict, random_results: defaultdict, no_bet_results: defaultdict, personality: str, output_dir: str):
    """Plot comparison of PPO agent vs Random and No Bet baselines."""
    plt.figure(figsize=(18, 6)) 
    
    strategy_names = ['PPO Trained', 'Random (Baseline)', 'Always No Bet']
    all_strategy_results = {
        strategy_names[0]: ppo_results,
        strategy_names[1]: random_results,
        strategy_names[2]: no_bet_results
    }
    
    metrics_to_plot = [
        ('final_balance', 'Final Balance Distribution', 'Balance ($)'),
        ('win_rate', 'Win Rate Distribution', 'Win Rate'),
        ('episode_reward', 'Episode Reward Distribution', 'Total Reward')
    ]

    for i, (metric_key, title, ylabel) in enumerate(metrics_to_plot):
        plt.subplot(1, len(metrics_to_plot), i + 1)
        data = [all_strategy_results[name][metric_key] for name in strategy_names]
        plt.boxplot(data, labels=strategy_names)
        plt.title(f'{title} ({personality} Personality)')
        plt.ylabel(ylabel)
        plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle(f'PPO vs Baselines for {personality} Personality ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")})', fontsize=16)
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(output_dir, f'ppo_vs_baselines_{personality}_{timestamp_str}.png') # New filename
    plt.savefig(plot_filename)
    plt.close()
    print(f"Plot saved to {plot_filename}")

def main():
    parser = argparse.ArgumentParser(description='Compare a trained PPO agent against random and no-bet baseline strategies.') # Updated desc
    parser.add_argument('--personality', type=str, default='standard', choices=['standard', 'thrill_seeker', 'loss_averse', 'play_for_time'], help='Gambler personality for the environment and model.')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the trained PPO model (.zip file). If not provided, path is inferred from personality.')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to run for each strategy.')
    args = parser.parse_args()

    model_path_to_load = args.model_path
    if model_path_to_load is None:
        model_path_to_load = f"personality_agents/{args.personality}/models/ppo_two_up_{args.personality}.zip"
        print(f"PPO model path not provided, inferring based on personality: {model_path_to_load}")

    if not os.path.exists(model_path_to_load):
        print(f"Error: PPO Model file not found at {model_path_to_load}")
        return

    print(f"Initializing environments with personality: {args.personality}")
    env_ppo = TwoUpEnvNumba(gambler_personality=args.personality)
    env_random = TwoUpEnvNumba(gambler_personality=args.personality)
    env_no_bet = TwoUpEnvNumba(gambler_personality=args.personality)
    
    print(f"Loading PPO model from: {model_path_to_load}")
    try:
        ppo_model = PPO.load(model_path_to_load, device="cpu")
    except Exception as e:
        print(f"Error loading PPO model: {e}")
        return
    
    print(f"\nRunning simulations for {args.episodes} episodes each (Personality: {args.personality})...")
    
    print("Running trained PPO agent...")
    ppo_results = run_simulation(env_ppo, model=ppo_model, num_episodes=args.episodes, strategy_name=f"PPO ({args.personality})")
    
    print("Running random agent...")
    random_results = run_simulation(env_random, model=None, num_episodes=args.episodes, strategy_name="Random Baseline")

    print("Running 'always no bet' agent...")
    no_bet_results = run_simulation(env_no_bet, model=None, num_episodes=args.episodes, always_no_bet=True, strategy_name="Always No Bet")

    report_dir = f"personality_agents/{args.personality}/reports"
    os.makedirs(report_dir, exist_ok=True)
    
    # Simplified JSON report for PPO vs Baselines
    ppo_vs_baselines_summary = {
        "report_timestamp": datetime.datetime.now().isoformat(),
        "comparison_personality_context": args.personality,
        "ppo_model_path": model_path_to_load,
        "num_episodes_per_strategy": args.episodes,
        "strategy_results": {
            "ppo_trained": {
                "average_final_balance": float(np.mean(ppo_results['final_balance'])),
                "std_final_balance": float(np.std(ppo_results['final_balance'])),
                "min_final_balance": float(np.min(ppo_results['final_balance'])),
                "max_final_balance": float(np.max(ppo_results['final_balance'])),
                "average_win_rate": float(np.mean(ppo_results['win_rate'])),
                "average_episode_reward": float(np.mean(ppo_results['episode_reward'])),
                "final_balance_raw_data": [float(b) for b in ppo_results['final_balance']]
            },
            "random_baseline": {
                "average_final_balance": float(np.mean(random_results['final_balance'])),
                "std_final_balance": float(np.std(random_results['final_balance'])),
                "min_final_balance": float(np.min(random_results['final_balance'])),
                "max_final_balance": float(np.max(random_results['final_balance'])),
                "average_win_rate": float(np.mean(random_results['win_rate'])),
                "average_episode_reward": float(np.mean(random_results['episode_reward'])),
                "final_balance_raw_data": [float(b) for b in random_results['final_balance']]
            },
            "always_no_bet": {
                "average_final_balance": float(np.mean(no_bet_results['final_balance'])),
                "std_final_balance": float(np.std(no_bet_results['final_balance'])),
                "min_final_balance": float(np.min(no_bet_results['final_balance'])),
                "max_final_balance": float(np.max(no_bet_results['final_balance'])),
                "average_win_rate": float(np.mean(no_bet_results['win_rate'])),
                "average_episode_reward": float(np.mean(no_bet_results['episode_reward'])),
                "final_balance_raw_data": [float(b) for b in no_bet_results['final_balance']]
            }
        }
    }
    
    print("\n--- PPO vs Baselines Summary ---")
    print(json.dumps(ppo_vs_baselines_summary, indent=4))

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = os.path.join(report_dir, f'ppo_vs_baselines_{args.personality}_{timestamp_str}.json') # New filename
    with open(report_filename, 'w') as f:
        json.dump(ppo_vs_baselines_summary, f, indent=4)
    print(f"\nPPO vs Baselines summary saved to: {report_filename}")
    
    plot_results(ppo_results, random_results, no_bet_results, args.personality, report_dir)

if __name__ == "__main__":
    main() 