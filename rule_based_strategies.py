# This file defines a library of rule-based (heuristic) strategies for the Two-Up game.
# It also provides functions to test these strategies within specific personality contexts by simulating game play using TwoUpEnvNumba.
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
import os
import datetime
import json
from collections import Counter

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

# Action definitions for clarity (assuming 1.0x is the middle multiplier)
ACTION_NO_BET = 0
ACTION_BET_HEADS_0_5X = 1
ACTION_BET_HEADS_1X = 2
ACTION_BET_HEADS_2X = 3
ACTION_BET_TAILS_0_5X = 4
ACTION_BET_TAILS_1X = 5
ACTION_BET_TAILS_2X = 6

# --- Individual Rule-Based Strategy Functions ---
def always_no_bet_strategy(observation, balance, initial_balance, base_bet_unit, action_space_obj=None):
    return ACTION_NO_BET, 0 # Bet amount placeholder is 0

def random_action_strategy(observation, balance, initial_balance, base_bet_unit, action_space_obj):
    # Samples an action from the provided action space (0-6)
    return action_space_obj.sample(), 0 # Bet amount placeholder is 0

def bet_heads_always_1x_strategy(observation, balance, initial_balance, base_bet_unit, action_space_obj=None):
    # Always bets 1.0x on Heads if possible
    if balance >= base_bet_unit * 1.0: # Check for 1x bet coverage
        return ACTION_BET_HEADS_1X, base_bet_unit * 1.0
    return ACTION_NO_BET, 0

def bet_tails_always_1x_strategy(observation, balance, initial_balance, base_bet_unit, action_space_obj=None):
    # Always bets 1.0x on Tails if possible
    if balance >= base_bet_unit * 1.0:
        return ACTION_BET_TAILS_1X, base_bet_unit * 1.0
    return ACTION_NO_BET, 0

def martingale_heads_strategy(observation, balance, initial_balance, base_bet_unit, last_bet_info):
    """Martingale: doubles bet on loss (up to 2x multiplier), resets on win. Bets on Heads.
    last_bet_info is a dict {\'won': bool, \'level_idx': int}
    Returns: action, bet_amount_for_display, updated_last_bet_info
    """
    bet_multipliers = np.array([0.5, 1.0, 2.0]) # Must match env
    
    current_level_idx = last_bet_info.get('level_idx', 0)
    last_bet_won = last_bet_info.get('won', True) # Assume first bet is like a win (resets)

    if last_bet_won:
        next_level_idx = 0 # Reset to smallest bet (0.5x)
    else:
        next_level_idx = min(current_level_idx + 1, len(bet_multipliers) - 1)

    chosen_multiplier = bet_multipliers[next_level_idx]
    actual_bet_amount_attempt = base_bet_unit * chosen_multiplier

    action_to_take = ACTION_NO_BET
    bet_amount_for_display = 0

    if balance >= actual_bet_amount_attempt and actual_bet_amount_attempt > 0.001: # Check can cover & positive bet
        if chosen_multiplier == 0.5: action_to_take = ACTION_BET_HEADS_0_5X
        elif chosen_multiplier == 1.0: action_to_take = ACTION_BET_HEADS_1X
        elif chosen_multiplier == 2.0: action_to_take = ACTION_BET_HEADS_2X
        # else: chosen_multiplier is somehow not in our list, defaults to NO_BET
        bet_amount_for_display = actual_bet_amount_attempt
    
    updated_last_bet_info = {'won': False, 'level_idx': next_level_idx} # Assume loss until proven win
    return action_to_take, bet_amount_for_display, updated_last_bet_info

STRATEGIES = {
    "Always No Bet": always_no_bet_strategy,
    "Random Action (New Space)": random_action_strategy,
    "Always Bet Heads (1x)": bet_heads_always_1x_strategy,
    "Always Bet Tails (1x)": bet_tails_always_1x_strategy,
    "Martingale Heads (Multi-level)": martingale_heads_strategy,
}

def run_strategy_test(strategy_func, strategy_name, num_episodes, 
                      initial_balance, base_bet_unit, max_episode_steps, 
                      gambler_personality='standard'):
    print(f"\nRunning strategy: {strategy_name} for {num_episodes} episodes...")
    env = TwoUpEnvNumba(
        initial_balance=initial_balance, 
        base_bet_amount=base_bet_unit, 
        gambler_personality=gambler_personality, 
        max_episode_steps=max_episode_steps
    )
    
    episode_rewards = []
    final_balances = []
    episode_lengths = []
    termination_reasons = []

    # Strategy-specific state persistence across steps in an episode
    martingale_last_bet_info = {'won': True, 'level_idx': 0}

    for _ in tqdm(range(num_episodes), desc=f"Testing {strategy_name}"):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward_ep = 0
        ep_len = 0
        
        if strategy_name == "Martingale Heads (Multi-level)":
            martingale_last_bet_info = {'won': True, 'level_idx': 0} # Reset for each episode
            
        while not terminated and not truncated:
            current_balance = env.balance # Get balance directly from env for strategies
            action_to_take = ACTION_NO_BET
            
            if strategy_name == "Random Action (New Space)":
                action_to_take, _ = strategy_func(obs, current_balance, initial_balance, base_bet_unit, env.action_space)
            elif strategy_name == "Martingale Heads (Multi-level)":
                action_to_take, _, martingale_last_bet_info = strategy_func(
                    obs, current_balance, initial_balance, base_bet_unit, martingale_last_bet_info
                )
            else:
                action_to_take, _ = strategy_func(obs, current_balance, initial_balance, base_bet_unit, None)

            prev_balance = env.balance
            obs, reward, terminated, truncated, info = env.step(action_to_take)
            total_reward_ep += reward
            ep_len += 1

            if strategy_name == "Martingale Heads (Multi-level)" and info.get('last_bet_amount', 0) > 0.001:
                if env.balance > prev_balance: # Won bet
                    martingale_last_bet_info['won'] = True
                elif env.balance < prev_balance: # Lost bet
                    martingale_last_bet_info['won'] = False
                # If balance is same (e.g. Odds or no bet), 'won' state doesn't change from its default (False for current bet)
                # but level_idx has already been updated for the *next* bet.

        episode_rewards.append(total_reward_ep)
        final_balances.append(env.balance)
        episode_lengths.append(ep_len)
        termination_reasons.append(info.get('termination_reason', 'unknown'))
    
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0
    avg_balance = np.mean(final_balances) if final_balances else initial_balance
    std_dev_balance = np.std(final_balances) if final_balances else 0
    min_balance = np.min(final_balances) if final_balances else initial_balance
    max_balance = np.max(final_balances) if final_balances else initial_balance
    avg_ep_length = np.mean(episode_lengths) if episode_lengths else 0
    
    reason_counts = Counter(termination_reasons)
    end_reason_distribution = {reason: count / num_episodes for reason, count in reason_counts.items()}

    summary = {
        "strategy_name": strategy_name,
        "avg_episode_reward": float(avg_reward),
        "avg_final_balance": float(avg_balance),
        "std_dev_final_balance": float(std_dev_balance),
        "min_final_balance": float(min_balance),
        "max_final_balance": float(max_balance),
        "avg_episode_length": float(avg_ep_length),
        "num_episodes": num_episodes,
        "end_reason_distribution": end_reason_distribution,
        "final_balance_raw_data": [float(b) for b in final_balances]
    }
    return summary

def main():
    parser = argparse.ArgumentParser(description="Run rule-based strategies on the TwoUp environment for universal baseline metrics.")
    parser.add_argument("--num-episodes", type=int, default=1000, help="Number of episodes for each strategy.")
    parser.add_argument("--initial-balance", type=float, default=100.0, help="Initial balance for the env.")
    parser.add_argument("--base-bet", type=float, default=10.0, help="Base bet unit for the env.")
    parser.add_argument("--max-episode-steps", type=int, default=200, help="Max steps per episode for the env.")
    args = parser.parse_args()

    all_strategy_summaries = []

    for name, func in STRATEGIES.items():
        summary = run_strategy_test(
            strategy_func=func, 
            strategy_name=name, 
            num_episodes=args.num_episodes, 
            initial_balance=args.initial_balance,
            base_bet_unit=args.base_bet, # Passed as base_bet_unit
            max_episode_steps=args.max_episode_steps,
            gambler_personality='standard' # Rule-based strategies use 'standard' context for rewards
        )
        all_strategy_summaries.append(summary)
    
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(reports_dir, f"rule_based_strategies_summary_universal_{timestamp}.json")
    
    if all_strategy_summaries:
        with open(filename, 'w') as f:
            json.dump(all_strategy_summaries, f, indent=4)
        print(f"\nAll rule-based strategy summaries saved to {filename}")
    else:
        print("No rule-based strategy summaries were generated.")

if __name__ == "__main__":
    main() 