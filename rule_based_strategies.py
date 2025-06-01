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
        pass # Further logic if needed upon termination
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

    return (
        balance, 
        total_reward, 
        rounds_played, 
        max_balance_seen, 
        min_balance_seen, 
        games_won_in_episode, 
        total_games_in_episode,
        end_reason_code
    )

def run_episode(strategy_func, initial_balance: int, base_bet: int, gambler_personality: str, max_rounds: int = 200) -> Dict:
    """Run a single episode using the provided strategy function with a TwoUpEnvNumba instance."""
    env = TwoUpEnvNumba(
        initial_balance=float(initial_balance), 
        base_bet_amount=base_bet, 
        gambler_personality=gambler_personality,
        max_episode_steps=max_rounds # Ensure env respects max_rounds
    )
    obs, info = env.reset()
    done = False
    episode_reward_sum = 0.0
    
    # Data to pass to Numba strategy funcs (some might not be directly in obs from env)
    # Numba strategies expect: obs, last_results, current_streak, base_bet
    # We need to ensure these are correctly mapped from the env state.

    for _current_round in range(max_rounds): # Loop with max_rounds explicitly
        # The Numba strategy functions expect specific inputs.
        # obs from env is a numpy array. last_results, current_streak, base_bet need to be correctly sourced.
        # Assuming obs from env.reset() and env.step() is compatible with what Numba funcs expect as first arg.
        # env.last_3_results and env.current_streak should be used.
        action_choice, bet_for_action = strategy_func(
            obs,                    # Current observation from the environment
            env.last_3_results,     # last_results from the environment
            env.current_streak,     # current_streak from the environment
            env.fixed_bet_amount    # Corrected: base_bet from the environment is env.fixed_bet_amount
        )
        
        # The Numba strategy functions might return a specific bet amount (bet_for_action),
        # but TwoUpEnvNumba.step() primarily takes an action type (0: no bet, 1: heads, 2: tails)
        # and then internally determines the actual bet stake based on env.base_bet_amount or personality-specific logic.
        # This creates a potential discrepancy if a rule-based strategy intends to bet an amount different from env.base_bet_amount.
        
        # Current handling:
        # 1. The action_choice (0, 1, or 2) from the strategy is prioritized.
        # 2. The bet_for_action returned by the strategy is largely ignored by env.step(), unless it's 0.
        # 3. If bet_for_action is 0 (indicating the strategy wants to bet $0) AND action_choice is a betting action (1 or 2),
        #    action_choice is overridden to 0 (no bet). This ensures a strategy can decide to not bet.
        if bet_for_action == 0 and action_choice != 0:
            action_choice = 0 # Force no-bet if strategy calculated a zero bet amount for a betting action.
        
        # TODO: Resolve how rule-based strategies can specify exact bet amounts if they differ from env.base_bet_amount.
        # Potential solutions:
        #   a) Modify TwoUpEnvNumba.step(action, optional_bet_amount=None): 
        #      If optional_bet_amount is provided, use it directly, bypassing internal base_bet/personality logic for stake size.
        #      This would be the most direct way for rule-based agents to control their exact bet stake.
        #   b) Introduce a new action space or wrapper for the environment when used with these rule-based agents
        #      that explicitly includes the bet amount if a bet is made.
        #   c) For strategies that *need* to vary their bet amount from base_bet (e.g., Streak Betting, Pattern Betting with half bets):
        #      Currently, streak_betting_strategy returns `base_bet * 2` or `base_bet // 2`.
        #      If env.step() doesn't use this, these strategies won't perform as designed regarding bet sizing.
        #      The `legend_strategy` (all-in) is a special case; its bet amount (current balance) must be respected.
        #      If `TwoUpEnvNumba` is modified (e.g. solution a), this section will need to pass `bet_for_action` to `env.step()`.

        obs, reward, terminated, truncated, current_info = env.step(action_choice)
        episode_reward_sum += reward
        done = terminated or truncated
        if done:
            break
            
    final_win_rate = np.float32(env.games_won / max(1, env.total_games)) if env.total_games > 0 else np.float32(0.0)
    # Get the termination reason from the last info dict or determine if max_rounds was hit without termination/truncation from env
    final_reason = current_info.get('reason')
    if not done and _current_round +1 >= max_rounds: # current_round is 0-indexed
        final_reason = 'max_steps_loop' # Distinguish from env's own truncation reason if loop ends it

    return {
        "final_balance": env.balance,
        "win_rate": final_win_rate,
        "episode_reward": episode_reward_sum,
        "total_games": env.total_games,
        "rounds_played": env.current_step_in_episode, # Use the env's step counter
        "end_reason": final_reason if final_reason else 'completed' # Add end_reason
    }

def run_strategy_test(scenario_name: str, strategy_func, base_bet: int, gambler_personality: str, num_episodes: int = 100, max_rounds_per_episode: int = 200) -> Tuple[defaultdict, Dict]:
    print(f"\nRunning test: {scenario_name} (Personality: {gambler_personality}, Base Bet: ${base_bet})")
    all_episode_details = []
    # Aggregate results similarly to how compare_personalities.py does for learned agents
    aggregated_results = defaultdict(list)
    initial_balance = 100 # Standard initial balance for these tests, can be parameter if needed

    for i in tqdm(range(num_episodes), desc=f"Simulating {scenario_name}"):
        episode_summary = run_episode(
            strategy_func=strategy_func, 
            initial_balance=initial_balance, 
            base_bet=base_bet, 
            gambler_personality=gambler_personality,
            max_rounds=max_rounds_per_episode
        )
        all_episode_details.append(episode_summary)
        aggregated_results['final_balance'].append(episode_summary['final_balance'])
        aggregated_results['win_rate'].append(episode_summary['win_rate'])
        aggregated_results['episode_reward'].append(episode_summary['episode_reward'])
        aggregated_results['total_games'].append(episode_summary['total_games'])
        aggregated_results['rounds_played'].append(episode_summary['rounds_played'])
        aggregated_results['end_reason'].append(episode_summary['end_reason']) # Aggregate end_reason

    # Summary statistics for this strategy
    # Calculate distribution of end reasons
    end_reason_counts = defaultdict(int)
    for reason in aggregated_results['end_reason']:
        end_reason_counts[reason if reason is not None else 'unknown'] += 1

    summary_stats = {
        "strategy_name": scenario_name,
        "personality_context": gambler_personality,
        "num_episodes": num_episodes,
        "base_bet": base_bet,
        "avg_final_balance": float(np.mean(aggregated_results['final_balance'])),
        "std_final_balance": float(np.std(aggregated_results['final_balance'])),
        "min_final_balance": float(np.min(aggregated_results['final_balance'])),
        "max_final_balance": float(np.max(aggregated_results['final_balance'])),
        "avg_win_rate": float(np.mean(aggregated_results['win_rate'])),
        "avg_episode_reward": float(np.mean(aggregated_results['episode_reward'])),
        "avg_total_games": float(np.mean(aggregated_results['total_games'])),
        "avg_rounds_played": float(np.mean(aggregated_results['rounds_played'])),
        "end_reason_distribution": dict(end_reason_counts),
        "final_balance_raw_data": [float(b) for b in aggregated_results['final_balance']]
    }
    print(f"Finished test for {scenario_name}. Avg Balance: ${summary_stats['avg_final_balance']:.2f}")
    # Return the aggregated_results for plotting, and summary_stats for JSON reporting
    return aggregated_results, summary_stats 

def test_env(num_episodes: int = 100, personality: str = 'standard'): # Added personality
    # Define test scenarios: (description, strategy_func, base_bet)
    # Base bet can be a common value, or derived from the personality context if needed.
    # For now, let's use a common base_bet for rule-based tests.
    common_base_bet = 10 

    scenarios = [
        ("Pattern Betting", pattern_betting_strategy, common_base_bet),
        ("Streak Betting", streak_betting_strategy, common_base_bet),
        ("Always Heads", always_heads_strategy, common_base_bet),
        ("Always Tails", always_tails_strategy, common_base_bet),
        ("Vibes Based (Random)", random_strategy, common_base_bet),
        ("Dickhead (No Bet/Conservative)", conservative_strategy, common_base_bet),
        ("Legend (All In Heads)", legend_strategy, common_base_bet) # Legend needs special handling for base_bet as it uses balance
    ]

    all_scenarios_summaries = []

    print(f"--- Testing Rule-Based Strategies (Personality Context: {personality}) ---")
    for desc, func, b_bet in scenarios:
        # Special handling for Legend strategy as its 'base_bet' is dynamic (all-in)
        # The base_bet passed to run_strategy_test for Legend might be ignored by the strategy itself.
        # We pass it for consistency, but the strategy will use current balance.
        _, strategy_summary = run_strategy_test(
            scenario_name=desc, 
            strategy_func=func, 
            base_bet=b_bet, 
            gambler_personality=personality, 
            num_episodes=num_episodes
        )
        all_scenarios_summaries.append(strategy_summary)
    
    print("\n--- Summary of Rule-Based Strategy Tests ---")
    # For tabulate, convert list of dicts to list of lists
    header = all_scenarios_summaries[0].keys() if all_scenarios_summaries else []
    rows = [list(s.values()) for s in all_scenarios_summaries]
    print(tabulate(rows, headers=header, tablefmt="grid"))

    # Save the summaries to a JSON file
    if all_scenarios_summaries: # Ensure there's data to save
        report_dir = "reports" # Save to a general reports directory
        os.makedirs(report_dir, exist_ok=True)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Filename no longer includes personality, as these are universal rule-based results
        report_filename = os.path.join(report_dir, f'rule_based_strategies_summary_universal_{timestamp_str}.json')
        
        output_json = {
            "report_timestamp": datetime.datetime.now().isoformat(),
            "report_type": "universal_rule_based_strategies", # Indicate type
            "num_episodes_per_strategy": num_episodes,
            "environment_context_for_rewards": personality, # Still note the personality used for *this specific run's rewards*
            "strategies": all_scenarios_summaries
        }
        with open(report_filename, 'w') as f:
            json.dump(output_json, f, indent=4)
        print(f"\nRule-based strategies summary saved to: {report_filename}")
    else:
        print("\nNo rule-based strategy summaries to save.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test various rule-based strategies for TwoUp to generate universal baseline metrics.') # Updated description
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to run for each strategy test.')
    # Personality argument is removed from CLI, but we'll run with a fixed one internally for rewards.
    # The key output metrics like final balance, win rate should be robust to this for rule-based agents.
    args = parser.parse_args()
    # Run with 'standard' personality for reward calculation consistency in this universal report.
    # The primary outcome metrics (balance, win_rate) are what we care about universally.
    test_env(num_episodes=args.episodes, personality='standard') 