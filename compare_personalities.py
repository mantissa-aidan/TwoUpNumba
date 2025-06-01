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
import time

# Action definitions for clarity (matching rule_based_strategies.py and env)
ACTION_NO_BET = 0
# ACTION_BET_HEADS_1X = 2 (example, if needed for a fixed strategy)
# ACTION_BET_TAILS_1X = 5 (example, if needed for a fixed strategy)

def evaluate_strategy(strategy_name, model_or_env, num_episodes, initial_balance, base_bet_unit, max_episode_steps, personality_for_env):
    """Evaluates a given strategy (PPO model, Random, or NoBet) over num_episodes."""
    print(f"Evaluating strategy: {strategy_name} for {num_episodes} episodes...")
    
    if strategy_name == "PPO Agent":
        eval_env = model_or_env # The model already has an env, but we might want to re-create for consistency
        # Re-create env for PPO to ensure consistent parameters are used for this evaluation run
        eval_env = TwoUpEnvNumba(
            initial_balance=initial_balance, 
            base_bet_amount=base_bet_unit, 
            max_episode_steps=max_episode_steps,
            gambler_personality=personality_for_env # Use the PPO agent's personality
        )
        model = model_or_env # This is actually the model itself
    else:
        # For Random and NoBet, model_or_env is the pre-created environment
        eval_env = TwoUpEnvNumba(
            initial_balance=initial_balance, 
            base_bet_amount=base_bet_unit, 
            max_episode_steps=max_episode_steps,
            gambler_personality=personality_for_env # For consistency, though action is fixed
        )
        model = None # No SB3 model for these

    final_balances = []
    episode_lengths = []
    total_rewards_per_episode = [] # Added to track episode rewards

    for i in tqdm(range(num_episodes), desc=f"Running {strategy_name}"):
        obs, info = eval_env.reset()
        terminated = False
        truncated = False
        current_episode_reward = 0.0 # Initialize for current episode
        ep_len = 0
        while not terminated and not truncated:
            if strategy_name == "PPO Agent":
                action, _ = model.predict(obs, deterministic=True)
            elif strategy_name == "Random (Baseline)":
                action = eval_env.action_space.sample() # Uses new Discrete(7) space
            elif strategy_name == "Always No Bet":
                action = ACTION_NO_BET # Use defined action for no bet
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")
            
            obs, reward, terminated, truncated, info = eval_env.step(action)
            current_episode_reward += reward # Accumulate reward
            ep_len +=1
        
        final_balances.append(eval_env.balance)
        episode_lengths.append(ep_len)
        total_rewards_per_episode.append(current_episode_reward) # Store total reward for the episode

    return {
        "strategy_name": strategy_name,
        "avg_final_balance": float(np.mean(final_balances)),
        "std_dev_final_balance": float(np.std(final_balances)),
        "min_final_balance": float(np.min(final_balances)),
        "max_final_balance": float(np.max(final_balances)),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "final_balance_raw_data": [float(b) for b in final_balances], # Ensure float for JSON
        "avg_total_episode_reward": float(np.mean(total_rewards_per_episode)), # Added
        "total_episode_reward_raw_data": [float(r) for r in total_rewards_per_episode] # Added
    }

def main():
    parser = argparse.ArgumentParser(description="Compare a PPO agent against baseline strategies.")
    parser.add_argument("--personality", type=str, required=True, choices=['standard', 'thrill_seeker', 'loss_averse', 'play_for_time'], help="Personality of the PPO agent to load.")
    parser.add_argument("--num-episodes", type=int, default=1000, help="Number of episodes for evaluation.")
    parser.add_argument("--initial-balance", type=float, default=100.0, help="Initial balance for the environment.")
    parser.add_argument("--base-bet", type=float, default=10.0, help="Base bet unit for the environment.")
    parser.add_argument("--max-episode-steps", type=int, default=200, help="Maximum steps per episode.")

    args = parser.parse_args()

    personality_path = f"personality_agents/{args.personality}"
    model_load_path = os.path.join(personality_path, "models", f"ppo_two_up_{args.personality}.zip")
    reports_save_path = os.path.join(personality_path, "reports")
    os.makedirs(reports_save_path, exist_ok=True)

    if not os.path.exists(model_load_path):
        print(f"Error: Model not found at {model_load_path}. Train the agent first.")
        return

    # Load PPO model
    ppo_model = PPO.load(model_load_path, device="cpu")
    print(f"Loaded PPO model for {args.personality} from {model_load_path}")

    all_results = []

    # Evaluate PPO Agent
    # The PPO agent's internal environment was set during its training.
    # For evaluation here, we create a new env with the specified CLI params for consistency.
    # The personality_for_env for PPO agent evaluation should be its own personality.
    ppo_results = evaluate_strategy(
        strategy_name="PPO Agent", 
        model_or_env=ppo_model, 
        num_episodes=args.num_episodes, 
        initial_balance=args.initial_balance, 
        base_bet_unit=args.base_bet,
        max_episode_steps=args.max_episode_steps,
        personality_for_env=args.personality
    )
    all_results.append(ppo_results)

    # Evaluate Random (Baseline)
    # For Random and NoBet, we pass the personality of the PPO agent being compared against
    # so that the environment context (rewards, if they subtly influence termination or observation for some reason)
    # is similar, though their actions are fixed/random.
    random_results = evaluate_strategy(
        strategy_name="Random (Baseline)", 
        model_or_env=None, # Env will be created inside evaluate_strategy
        num_episodes=args.num_episodes,
        initial_balance=args.initial_balance, 
        base_bet_unit=args.base_bet,
        max_episode_steps=args.max_episode_steps,
        personality_for_env=args.personality 
    )
    all_results.append(random_results)

    # Evaluate Always No Bet
    no_bet_results = evaluate_strategy(
        strategy_name="Always No Bet", 
        model_or_env=None, # Env will be created inside evaluate_strategy
        num_episodes=args.num_episodes, 
        initial_balance=args.initial_balance, 
        base_bet_unit=args.base_bet,
        max_episode_steps=args.max_episode_steps,
        personality_for_env=args.personality
    )
    all_results.append(no_bet_results)

    # Save results to JSON
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    json_filename = os.path.join(reports_save_path, f"ppo_vs_baselines_{args.personality}_{timestamp}.json")
    with open(json_filename, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"PPO vs Baselines comparison results saved to {json_filename}")

    # Plotting (Average Final Balance)
    strategy_names = [res["strategy_name"] for res in all_results]
    avg_final_balances = [res["avg_final_balance"] for res in all_results]
    std_devs = [res["std_dev_final_balance"] for res in all_results]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(strategy_names, avg_final_balances, yerr=std_devs, capsize=5, color=['blue', 'green', 'red'])
    plt.ylabel('Average Final Balance')
    plt.title(f'PPO ({args.personality}) vs Baselines: Avg Final Balance (after {args.num_episodes} episodes)')
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + np.sign(yval)*0.05*max(avg_final_balances), f'{yval:.2f}', ha='center', va='bottom' if yval >=0 else 'top')

    plot_filename = os.path.join(reports_save_path, f"ppo_vs_baselines_{args.personality}_{timestamp}.png")
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    # plt.show() # Optional: display plot

    # Plotting (Average Total Episode Reward) - NEW PLOT
    avg_episode_rewards = [res.get("avg_total_episode_reward", 0) for res in all_results] # Use .get for safety if key missing in older data

    plt.figure(figsize=(10, 6))
    reward_bars = plt.bar(strategy_names, avg_episode_rewards, capsize=5, color=['purple', 'orange', 'cyan'])
    plt.ylabel('Average Total Episode Reward')
    plt.title(f'PPO ({args.personality}) vs Baselines: Avg Total Episode Reward (after {args.num_episodes} episodes)')
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    for bar in reward_bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + np.sign(yval)*0.05*max(avg_episode_rewards, default=0), f'{yval:.2f}', ha='center', va='bottom' if yval >=0 else 'top')
    
    reward_plot_filename = os.path.join(reports_save_path, f"ppo_vs_baselines_ep_reward_{args.personality}_{timestamp}.png")
    plt.savefig(reward_plot_filename)
    print(f"Episode reward plot saved to {reward_plot_filename}")

if __name__ == "__main__":
    main() 