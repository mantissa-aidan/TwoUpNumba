import multiprocessing as mp
from two_up_env_numba import TwoUpEnvNumba, toss_coins_numba
from tabulate import tabulate
from typing import List, Tuple, Dict
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import time
from functools import partial
import argparse
from numba import jit, int32, float32
import os # Added for model path checking

from stable_baselines3 import PPO
import torch

# Global variable to store the loaded PPO model
PPO_AGENT_MODEL = None
MODEL_PATH = "trained_models/ppo_two_up"

# Bet amounts corresponding to PPO actions 0-5
PPO_BET_AMOUNTS = np.array([0, 5, 10, 20, 50, 100], dtype=np.int32)

def load_global_ppo_model():
    global PPO_AGENT_MODEL
    global MODEL_PATH # Ensure we can modify it if needed (e.g. if arg sets full .zip path)

    if PPO_AGENT_MODEL is not None:
        return PPO_AGENT_MODEL

    # Construct the full path to the .zip file if MODEL_PATH is just the base
    model_zip_path = MODEL_PATH if MODEL_PATH.endswith(".zip") else MODEL_PATH + ".zip"

    print(f"Attempting to load PPO model from: {model_zip_path}") # Debug print

    if os.path.exists(model_zip_path):
        print(f"Model file found at {model_zip_path}. Loading...")
        try:
            PPO_AGENT_MODEL = PPO.load(MODEL_PATH, device='cpu') # PPO.load can often handle basename
            print("PPO model loaded successfully.")
        except Exception as e:
            print(f"Error loading PPO model with base path '{MODEL_PATH}': {e}. Attempting with explicit .zip path...")
            try:
                PPO_AGENT_MODEL = PPO.load(model_zip_path, device='cpu') # Try with explicit .zip
                print("PPO model loaded successfully with explicit .zip path.")
            except Exception as e2:
                print(f"Error loading PPO model with explicit path '{model_zip_path}': {e2}. PPO Agent strategy will not run.")
                PPO_AGENT_MODEL = None
    else:
        print(f"PPO model file not found at {model_zip_path}. Searched based on MODEL_PATH: '{MODEL_PATH}'. PPO Agent strategy will not run.")
        PPO_AGENT_MODEL = None
    
    return PPO_AGENT_MODEL

# This function will NOT be @jit(nopython=True)
# It serves as the strategy function for the PPO agent, compatible with run_episode_numba_pure expects.
def ppo_agent_strategy(obs_from_numba_loop, last_3_results_numba, current_streak_numba, base_bet_ignored):
    # obs_from_numba_loop is: [rounds_played_count, last_3_r[0], last_3_r[1], last_3_r[2], 0, balance]
    # PPO model was trained on obs: [current_streak, last_3_r[0], last_3_r[1], last_3_r[2], current_bet (0 here), balance]

    model_to_use = PPO_AGENT_MODEL # Access the globally loaded model

    if model_to_use is None:
        # This might happen if loading failed or if called in a context where the global isn't set (e.g. fresh multiprocess worker)
        # For sequential execution or correct global model loading, this path should ideally not be hit often.
        # print("PPO_AGENT_MODEL not available in ppo_agent_strategy. Defaulting to No Bet.")
        return 0, 0.0 # No bet, 0 amount

    balance = obs_from_numba_loop[5]
    
    # Construct the observation for PPO model
    ppo_observation = np.array([
        float(current_streak_numba),
        float(last_3_results_numba[0]),
        float(last_3_results_numba[1]),
        float(last_3_results_numba[2]),
        0.0, # current_bet placeholder for PPO observation (always 0 before PPO decides its bet amount)
        float(balance)
    ], dtype=np.float32)

    try:
        ppo_action_raw, _ = model_to_use.predict(ppo_observation, deterministic=True)
        ppo_action = int(ppo_action_raw)
    except Exception as e:
        # print(f"Error during PPO model prediction: {e}. Defaulting to No Bet.")
        return 0, 0.0

    # Translate PPO action (0-6 from TwoUpEnvNumba action_space) 
    # to Numba loop's (action_choice, bet_amount_value)
    # Numba loop action_choice: 0=NoBet, 1=Heads, 2=Tails, 6=Quit

    if ppo_action == 6: # PPO Quit
        return 6, 0.0 # Numba loop Quit, 0 amount
    
    # PPO actions 0-5 correspond to indices in PPO_BET_AMOUNTS
    if not (0 <= ppo_action < len(PPO_BET_AMOUNTS)):
        # print(f"Warning: PPO action {ppo_action} out of bounds for PPO_BET_AMOUNTS. Defaulting to No Bet.")
        return 0, 0.0
        
    bet_amount_value = float(PPO_BET_AMOUNTS[ppo_action])

    if ppo_action == 0: # PPO No Bet (maps to bet_amount_value = 0)
        return 0, 0.0 # Numba loop NoBet, 0 amount
    elif ppo_action == 1: # PPO Env Action 1: Bet $5 on Heads
        return 1, bet_amount_value # Numba loop: 1=Heads, amount
    elif ppo_action == 2: # PPO Env Action 2: Bet $10 on Tails
        return 2, bet_amount_value # Numba loop: 2=Tails, amount
    elif ppo_action >= 3 and ppo_action <= 5: # PPO Env Actions 3,4,5: Bet $20,$50,$100 (on undefined target in env)
        # For this test loop, we map to "Bet on Heads" with the corresponding amount.
        return 1, bet_amount_value # Numba loop: 1=Heads, amount
    else:
        # This case should be covered by the bounds check or previous conditions.
        # print(f"Warning: Unhandled PPO action {ppo_action} after mapping. Defaulting to No Bet.")
        return 0, 0.0

# Strategy descriptions for layman's terms
STRATEGY_DESCRIPTIONS = {
    'Pattern Betting': """
    This strategy looks for patterns in the coin tosses:
    - If we see two tails in a row, bet full amount on heads
    - If we see two heads in a row, bet full amount on tails
    - If we see alternating results (heads-tails or tails-heads), bet half amount on the opposite of last result
    - If no clear pattern emerges after 50 rounds, quit the game
    
    The base bet amount ($10 or $20) determines your betting size:
    For example, with a $10 base:
    - Two tails → bet $10 on heads
    - Two heads → bet $10 on tails
    - Alternating results → bet $5 on the opposite of last result
    - After 50 rounds without pattern → quit
    """,
    
    'Streak Betting': """
    This strategy changes bet size based on winning streaks:
    - After winning twice in a row, double your bet
    - After losing, cut your bet in half
    - Otherwise, bet your normal amount
    The idea is to bet more when you're "hot" and less when you're "cold".
    
    The base bet amount ($10 or $20) is your starting bet size:
    For example, with a $10 base:
    - Normal bet: $10
    - After two wins: $20 (double)
    - After a loss: $5 (half)
    - Minimum bet is $5
    """,

    'Always Heads': """
    The simplest strategy - always bet on heads:
    - Every round, bet the base amount on heads
    - No pattern recognition, no streak management
    - Just pure heads-or-nothing betting
    """,

    'Always Tails': """
    The mirror of Always Heads:
    - Every round, bet the base amount on tails
    - No pattern recognition, no streak management
    - Just pure tails-or-nothing betting
    """,

    'Vibes Based': """
    Completely random betting:
    - Randomly choose between heads, tails, or no bet
    - Randomly choose bet amount
    - No strategy, just pure chance
    """,

    'Dickhead (No Bet)': """
    The safest strategy - never bet:
    - Keep your starting $100
    - Never risk any money
    - Guaranteed to end with exactly what you started with
    """,

    'Legend (All In Heads)': """
    The highest risk, highest (potential) reward strategy:
    - Always bet your entire current balance on Heads.
    - If you win, you (at least) double your money.
    - If you lose, you're out.
    - Live by the sword, die by the sword!
    """
}

@jit(nopython=True)
def pattern_betting_strategy(obs, last_results, current_streak, base_bet):
    """Bet based on patterns in last results, after tosses always occur."""
    rounds_played = obs[0]
    
    # Ensure we have at least two valid historical results (0 for Heads, 1 for Tails)
    # to check for patterns. Ignore Odds (2) or uninitialized (-1).
    last_toss = last_results[-1]
    second_last_toss = last_results[-2]

    pattern_found = False
    action_to_take = 0  # Default to no bet
    bet_for_action = 0

    # Check for patterns only if the last two tosses were definitive (Heads or Tails)
    if (last_toss == 0 or last_toss == 1) and (second_last_toss == 0 or second_last_toss == 1):
        # If we see 2 tails in a row (1, 1), bet full on heads (action 1)
        if second_last_toss == 1 and last_toss == 1:
            action_to_take = 1  # Bet on heads
            bet_for_action = base_bet
            pattern_found = True
        # If we see 2 heads in a row (0, 0), bet full on tails (action 2)
        elif second_last_toss == 0 and last_toss == 0:
            action_to_take = 2  # Bet on tails
            bet_for_action = base_bet
            pattern_found = True
        # If we see alternating results (H-T or T-H), bet half on the opposite of last result
        elif last_toss != second_last_toss: # This implicitly means 0,1 or 1,0
            action_to_take = (2 if last_toss == 0 else 1) # Bet opposite of last_toss
            bet_for_action = base_bet // 2 if base_bet > 0 else 0
            pattern_found = True

    # If we've played more than 50 rounds AND no clear pattern emerged in this round, quit.
    if rounds_played > 50 and not pattern_found:
        return 6, 0  # Quit action

    # If a pattern was found, and base_bet is > 0, return the action.
    # Otherwise, (no pattern or base_bet is 0), default to no bet.
    if pattern_found and base_bet > 0:
        return action_to_take, bet_for_action
    else:
        return 0, 0 # No bet / conservative base bet

@jit(nopython=True)
def streak_betting_strategy(obs, last_results, current_streak, base_bet):
    """Bet based on current winning streak."""
    if base_bet == 0:  # For conservative strategy
        return 0, 0
    if current_streak >= 2:
        # On a winning streak, increase bet
        return 1, base_bet * 2  # Bet on heads with double amount
    elif current_streak == 0:
        # After a loss, decrease bet
        return 1, max(5, base_bet // 2)  # Bet on heads with half amount
    else:
        # Normal bet
        return 1, base_bet

@jit(nopython=True)
def always_heads_strategy(obs, last_results, current_streak, base_bet):
    """Always bet on heads."""
    if base_bet == 0:  # For conservative strategy
        return 0, 0
    return 1, base_bet

@jit(nopython=True)
def always_tails_strategy(obs, last_results, current_streak, base_bet):
    """Always bet on tails."""
    if base_bet == 0:  # For conservative strategy
        return 0, 0
    return 2, base_bet

@jit(nopython=True)
def random_strategy(obs, last_results, current_streak, base_bet):
    """Random betting strategy."""
    action = np.random.randint(0, 3)  # 0: no bet, 1: heads, 2: tails
    if action == 0:  # If no bet, return 0 bet amount
        return action, 0
    # Randomly choose between base_bet and half base_bet
    bet_amount = base_bet if np.random.random() > 0.5 else max(5, base_bet // 2)
    return action, bet_amount

@jit(nopython=True)
def conservative_strategy(obs, last_results, current_streak, base_bet):
    """Never bet - keep your money."""
    return 0, 0

@jit(nopython=True)
def legend_strategy(obs, last_results, current_streak, base_bet):
    """Always go all-in on Heads."""
    current_balance = obs[5]
    if current_balance <= 0:
        return 0, 0 # No bet if no money
    
    # Action 1 for Heads, bet entire current balance
    return 1, current_balance

@jit(nopython=True)
def _perform_numba_game_step(
    action_choice: int,      # 0: no bet, 1: heads, 2: tails
    bet_amount_value: float,
    balance: float,
    current_streak: int,
    last_3_results_input: np.ndarray, # dtype=np.int32
    games_won: int,
    total_games: int
):
    """
    Performs a single step of the Two-Up game.
    Returns: (reward, new_balance, new_current_streak, new_last_3_results, 
              new_games_won, new_total_games, terminated, toss_outcome_for_history)
    toss_outcome_for_history: 0=Heads, 1=Tails, 2=Odds, -1=No Toss, -2=Invalid Bet
    """
    reward = 0.0
    new_balance = balance
    new_current_streak = current_streak
    new_games_won = games_won
    new_total_games = total_games
    new_last_3_results = last_3_results_input.copy()
    toss_outcome_for_history = -1 # Default to no toss

    # Handle invalid bet (betting more than available an actual amount)
    if (action_choice == 1 or action_choice == 2) and bet_amount_value > 0 and bet_amount_value > new_balance:
        reward = -10.0 # Penalty for attempting to bet more than balance
        toss_outcome_for_history = -2 # Indicate invalid bet
        terminated = new_balance <= 0 
        return (reward, new_balance, new_current_streak, new_last_3_results,
                new_games_won, new_total_games, terminated, toss_outcome_for_history)

    # Always perform a coin toss for the round if we proceed beyond an invalid bet check
    coin1 = np.random.randint(0, 2) # 0 for heads, 1 for tails
    coin2 = np.random.randint(0, 2)
    actual_toss_result = -1 # 0: HH (Heads), 1: TT (Tails), 2: HT/TH (Odds)
    if coin1 == coin2:
        actual_toss_result = 0 if coin1 == 0 else 1
    else:
        actual_toss_result = 2 # Odds
    
    toss_outcome_for_history = actual_toss_result

    # Update last 3 results: [oldest, middle, newest]
    new_last_3_results[0] = new_last_3_results[1]
    new_last_3_results[1] = new_last_3_results[2]
    new_last_3_results[2] = actual_toss_result

    if action_choice == 0 or bet_amount_value == 0:  # No bet placed or zero amount bet
        reward = 0.0
        # Game state like streak, balance (from betting) doesn't change due to player action.
        # total_games and games_won also don't change as no bet was resolved.
    else: # Bet on heads (action_choice == 1) or tails (action_choice == 2) with a valid amount
        if actual_toss_result != 2:  # Not odds (i.e., a definitive Heads or Tails outcome)
            new_total_games += 1
            won_bet = (action_choice == 1 and actual_toss_result == 0) or \
                      (action_choice == 2 and actual_toss_result == 1)

            if won_bet:
                reward = bet_amount_value
                new_balance += bet_amount_value
                new_current_streak += 1
                new_games_won += 1
            else: # Lost bet
                reward = -bet_amount_value
                new_balance -= bet_amount_value
                new_current_streak = 0 # Reset streak
        else: # Odds - bet is returned, no win/loss
            reward = 0.0 # No win/loss, no change to streak or total_games on "Odds"

    terminated = new_balance <= 0
    if terminated and toss_outcome_for_history != -2: # If game play (not invalid bet penalty) caused termination
        reward = -100.0 # Standard penalty for going broke

    return (reward, new_balance, new_current_streak, new_last_3_results,
            new_games_won, new_total_games, terminated, toss_outcome_for_history)

@jit(nopython=True)
def run_episode_numba_pure(strategy_func, initial_balance: float, base_bet: int, max_rounds: int):
    """Numba-optimized episode runner, operating on pure Numba-compatible types."""
    balance = initial_balance
    total_reward = 0.0
    rounds_played = 0
    max_balance_seen = initial_balance
    min_balance_seen = initial_balance
    current_streak = 0
    last_3_results = np.full(3, -1, dtype=np.int32) 
    games_won_in_episode = 0
    total_games_in_episode = 0
    end_reason_code = 0 
    invalid_bet_attempts = 0
    obs_placeholder = np.zeros(6, dtype=np.float32)
    last_bet_amount_value_for_obs = 0.0 # Track previous bet for obs

    while rounds_played < max_rounds:
        obs_placeholder[0] = float(rounds_played)
        obs_placeholder[1] = float(last_3_results[0])
        obs_placeholder[2] = float(last_3_results[1])
        obs_placeholder[3] = float(last_3_results[2])
        obs_placeholder[4] = last_bet_amount_value_for_obs # Previous bet amount
        obs_placeholder[5] = balance # Current balance
        
        action_choice, bet_amount_value = strategy_func(obs_placeholder, last_3_results, current_streak, base_bet)

        # Update last_bet_amount_value_for_obs for the *next* iteration's obs
        if action_choice == 1 or action_choice == 2: # Bet on H or T
            last_bet_amount_value_for_obs = float(bet_amount_value)
        else: # No bet or Quit
            last_bet_amount_value_for_obs = 0.0

        if action_choice == 6:
            end_reason_code = 2
            break

        if (action_choice == 1 or action_choice == 2) and bet_amount_value == 0:
            action_choice = 0

        reward_from_step, new_balance, new_streak, new_last_3, \
        new_games_won, new_total_games, terminated, toss_hist_val = _perform_numba_game_step(
            action_choice, float(bet_amount_value), balance, current_streak, 
            last_3_results, games_won_in_episode, total_games_in_episode
        )

        if toss_hist_val == -2:
            invalid_bet_attempts += 1
            if invalid_bet_attempts >= 3:
                end_reason_code = 3 
                break
            total_reward += reward_from_step 
            balance = new_balance
            min_balance_seen = min(min_balance_seen, balance)
            max_balance_seen = max(max_balance_seen, balance)
            if balance <= 0:
                end_reason_code = 1
                break
            continue

        invalid_bet_attempts = 0
        total_reward += reward_from_step
        balance = new_balance
        current_streak = new_streak
        last_3_results = new_last_3
        games_won_in_episode = new_games_won
        total_games_in_episode = new_total_games
        
        rounds_played += 1
        min_balance_seen = min(min_balance_seen, balance)
        max_balance_seen = max(max_balance_seen, balance)
        
        if terminated:
            end_reason_code = 1
            break
            
    if rounds_played >= max_rounds and end_reason_code == 0:
        end_reason_code = 0

    win_rate = (games_won_in_episode / float(total_games_in_episode)) if total_games_in_episode > 0 else 0.0
    
    end_reason_map = {
        0: 'max_rounds',
        1: 'broke',
        2: 'voluntary_quit',
        3: 'invalid_bet_loop'
    }
    end_reason_str = end_reason_map.get(end_reason_code, 'unknown')

    return (
        balance, 
        total_reward, 
        rounds_played, 
        max_balance_seen, 
        min_balance_seen, 
        games_won_in_episode, 
        total_games_in_episode,
        end_reason_code,
        win_rate
    )

def run_strategy_test(scenario_name: str, strategy_func, base_bet: int, num_episodes: int = 100, ppo_model_unused=None, ppo_env_for_obs_unused=None) -> Tuple[Dict, List[Dict], Dict]:
    all_final_balances = []
    all_total_rewards = []
    all_rounds_played = []
    all_max_balances = []
    all_win_rates = []
    episode_details_list = [] 
    end_reasons_counts = defaultdict(int)

    results_for_episodes = []
    for _ in tqdm(range(num_episodes), desc=f"Testing {scenario_name}", leave=False):
        # base_bet is ignored by ppo_agent_strategy, which uses PPO_BET_AMOUNTS
        episode_result = run_episode_numba_pure(strategy_func, 100.0, base_bet, 200) 
        results_for_episodes.append(episode_result)
        end_reasons_counts[episode_result[-2]] += 1

    for res in results_for_episodes:
        all_final_balances.append(res[0])
        all_total_rewards.append(res[1])
        all_rounds_played.append(res[2])
        all_max_balances.append(res[3])
        all_win_rates.append(res[8])
        episode_details_list.append({
            'final_balance': res[0],
            'total_reward': res[1],
            'rounds_played': res[2],
            'max_balance': res[3],
            'min_balance': res[4],
            'games_won_in_episode': res[5],
            'total_decisive_games_in_episode': res[6],
            'end_reason': res[-2],
            'win_rate': res[8]
        })

    avg_final_balance = np.mean(all_final_balances) if all_final_balances else 0
    avg_total_reward = np.mean(all_total_rewards) if all_total_rewards else 0
    avg_rounds_played = np.mean(all_rounds_played) if all_rounds_played else 0
    avg_max_balance = np.mean(all_max_balances) if all_max_balances else 0
    avg_win_rate = np.mean(all_win_rates) if all_win_rates else 0
    
    summary = {
        "Strategy": scenario_name,
        "Avg Final Balance": f"${avg_final_balance:.2f}",
        "Avg Total Reward": f"${avg_total_reward:.2f}",
        "Avg Rounds": f"{avg_rounds_played:.1f}",
        "Avg Win Rate": f"{avg_win_rate:.1%}",
        "Avg Max Balance": f"${avg_max_balance:.2f}",
    }
    return summary, episode_details_list, end_reasons_counts

def test_rl_agent_only(num_episodes: int = 100):
    load_global_ppo_model() 

    if PPO_AGENT_MODEL is None:
        print("PPO model could not be loaded. Exiting.")
        return

    strategy_name = "PPO Agent"
    # base_bet_for_ppo is a dummy value, as ppo_agent_strategy ignores it.
    ppo_strategy_func = partial(ppo_agent_strategy, base_bet_ignored=0)

    print(f"\nRunning Test for: {strategy_name}")
    print(f"Note: Each episode has a maximum of {200} rounds")
    print("=" * 70)
    
    start_time = time.time()

    summary, episode_details, end_reasons = run_strategy_test(strategy_name, ppo_strategy_func, 0, num_episodes)
    
    end_time = time.time()
    print(f"\nTotal testing time for {strategy_name}: {end_time - start_time:.2f} seconds\n")

    # Print strategy description for PPO Agent
    if strategy_name in STRATEGY_DESCRIPTIONS:
        print("Strategy Description:")
        print("=" * 70 + "\n")
        print(f"{strategy_name}:")
        print(STRATEGY_DESCRIPTIONS[strategy_name])
        print("-" * 30 + "\n")
    
    # Print total rounds for the PPO agent strategy
    print(f"Strategy Totals for {strategy_name}:")
    print("=" * 70 + "\n")
    total_rounds_for_strategy = sum(e['rounds_played'] for e in episode_details)
    avg_rounds_from_summary = float(summary["Avg Rounds"])
    print(f"  Total Episodes: {num_episodes}")
    print(f"  Total Rounds: {total_rounds_for_strategy}")
    print(f"  Average Rounds per Episode: {avg_rounds_from_summary:.1f}\n")

    print(f"RL Agent ({strategy_name}) Performance Analysis")
    print("=" * 70)
    print(f"Results averaged over {num_episodes} episodes\n")
    
    headers = ["Strategy", "Avg Final Balance", "Avg Total Reward", "Avg Rounds", "Avg Win Rate", "Avg Max Balance"]
    table_data = [[summary.get(key, "N/A") for key in headers]] # Create a list containing one list for the single strategy
    
    print("Strategy Results:")
    print(tabulate(table_data, headers=headers, tablefmt="heavy_grid"))
    print("\n")

    print("Episode End Reasons:")
    for reason, count in end_reasons.items():
        print(f"   {reason}: {count} episodes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run tests for the trained PPO RL agent.')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to run for the RL agent.')
    parser.add_argument('--model_path', type=str, default="trained_models/ppo_two_up", help='Path to the trained PPO model.') # Added model_path arg
    args = parser.parse_args()
    
    # Update global MODEL_PATH if provided via CLI
    if args.model_path:
        MODEL_PATH = args.model_path

    test_rl_agent_only(num_episodes=args.episodes)