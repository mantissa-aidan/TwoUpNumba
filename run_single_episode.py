import numpy as np
from stable_baselines3 import PPO
from two_up_env_numba import TwoUpEnvNumba
import argparse
import torch # For device checking, though we force CPU for this script
import os # Added for os.path.exists

def run_single_episode_with_details(env, model, max_steps=200):
    """Run a single episode with a trained model and print step-by-step details."""
    print(f"--- Starting Single Episode Simulation (Max Steps: {max_steps}, Personality: {env.gambler_personality}) ---")
    obs, info = env.reset()
    done = False
    truncated_episode = False # To specifically track truncation due to max_steps
    episode_reward = 0.0
    step_count = 0

    while not done:
        step_count += 1
        action, _ = model.predict(obs, deterministic=True)
        
        # Get current state before step for more comprehensive logging
        current_balance_before_step = env.balance
        current_streak_before_step = env.current_streak
        last_3_results_before_step = env.last_3_results.copy() # Copy to avoid mutation issues before printing
        current_bet_before_step = env.current_bet

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        action_meaning = "No Bet"
        if action == 1:
            action_meaning = "Bet Heads"
        elif action == 2:
            action_meaning = "Bet Tails"

        result_meaning = "N/A (No bet or pre-toss)"
        if info.get('result') is not None:
            if info['result'] == 0: result_meaning = "Heads"
            elif info['result'] == 1: result_meaning = "Tails"
            elif info['result'] == 2: result_meaning = "Odds"

        print(f"\nStep: {step_count}")
        print(f"  Observation: {obs}")
        print(f"  Action Chosen: {action} ({action_meaning})")
        print(f"  Bet Amount Used by Env: {env.current_bet if action !=0 else 0}") # Shows the actual bet processed
        print(f"  Coin Toss Result: {result_meaning} (Raw: {info.get('result')})")
        print(f"  Reward for this step: {reward:.2f}") # Format reward
        print(f"  Balance: ${env.balance:.2f}") # Use env.balance directly for final values
        print(f"  Current Streak: {env.current_streak}") # env object directly
        print(f"  Last 3 Results: {env.last_3_results}")
        
        done = terminated or truncated

        if not done and step_count >= max_steps:
            print(f"\nEpisode truncated after reaching max steps: {max_steps}")
            done = True
            truncated_episode = True # Mark that truncation was due to step limit
            # Update info for this specific truncation if necessary, though env might do it
            if 'reason' not in info or info['reason'] is None:
                 info['reason'] = 'max_steps_reached'

        if done:
            print("\n--- Episode Finished ---")
            reason = info.get('reason', 'Unknown')
            if truncated_episode and reason == 'max_steps_reached':
                print(f"  Reason for termination: Max steps ({max_steps}) reached.")
            elif reason:
                print(f"  Reason for termination: {reason}")
            else:
                print(f"  Reason for termination: Terminal state reached or externally truncated.")

            print(f"  Final Balance: ${env.balance:.2f}") # Use env.balance directly for final values
            print(f"  Total Episode Reward: {episode_reward:.2f}")
            # Ensure total_games and win_rate are correct after truncation
            total_games_played = env.total_games
            win_rate_episode = (env.games_won / max(1, total_games_played)) if total_games_played > 0 else 0.0
            print(f"  Total Games Played in Episode: {total_games_played}")
            print(f"  Overall Win Rate for Episode: {win_rate_episode:.2%}")

def main():
    parser = argparse.ArgumentParser(description='Run a single episode with a trained PPO agent and print details.')
    parser.add_argument('--model_path', type=str, required=False, default=None, help='Path to the trained PPO model (.zip file). If not provided, path is inferred from personality.')
    parser.add_argument('--max_steps', type=int, default=200, help='Maximum number of steps to run in the episode.')
    parser.add_argument('--personality', type=str, default='standard', choices=['standard', 'thrill_seeker', 'loss_averse', 'play_for_time'], help='Gambler personality for the environment context. Also used to infer model path if not provided.')
    args = parser.parse_args()

    model_path_to_load = args.model_path
    if model_path_to_load is None:
        model_path_to_load = f"personality_agents/{args.personality}/models/ppo_two_up_{args.personality}.zip"
        print(f"Model path not provided, inferring based on personality: {model_path_to_load}")

    if not os.path.exists(model_path_to_load):
        print(f"Error: Model file not found at {model_path_to_load}")
        print("Please ensure the path is correct or that a model for the specified personality has been trained.")
        return

    print(f"Loading model from: {model_path_to_load}")
    device = torch.device("cpu") 
    
    try:
        model = PPO.load(model_path_to_load, device=device)
    except Exception as e:
        print(f"Error loading the model: {e}")
        print("Please ensure the model path is correct and the model was saved by Stable Baselines3 PPO.")
        return

    print(f"Initializing environment with personality: {args.personality}")
    env = TwoUpEnvNumba(render_mode=None, gambler_personality=args.personality)

    run_single_episode_with_details(env, model, max_steps=args.max_steps)

if __name__ == "__main__":
    main() 