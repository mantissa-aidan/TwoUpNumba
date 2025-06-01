from two_up_env import TwoUpEnv
from tabulate import tabulate
from typing import List, Tuple, Dict
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import time
import multiprocessing as mp
from functools import partial
import argparse

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
    """
}

def run_episode(env: TwoUpEnv, strategy_func, base_bet: int, pbar: tqdm = None, max_rounds: int = 200) -> Dict:
    """Run a single episode with a dynamic betting strategy."""
    obs, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    rounds_played = 0
    max_balance = 100  # Starting balance
    min_balance = 100
    current_streak = 0
    last_results = [0, 0, 0]  # Track last 3 results
    last_update_time = time.time()
    last_pbar_update = 0
    
    while not (terminated or truncated) and rounds_played < max_rounds:
        # Check for timeout
        if time.time() - last_update_time > 5:  # 5 second timeout
            print(f"\nEpisode timed out after {rounds_played} rounds")
            break
            
        # Get action and bet amount from strategy
        action, bet_amount = strategy_func(obs, last_results, current_streak, base_bet)
        
        # Place bet
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rounds_played += 1
        max_balance = max(max_balance, info['balance'])
        min_balance = min(min_balance, info['balance'])
        
        # Update streak and results
        if reward > 0:
            current_streak += 1
        else:
            current_streak = 0
        last_results = obs[1:4]  # Update last 3 results from observation
        
        # Update progress bar less frequently (every 100 rounds)
        if pbar is not None and rounds_played - last_pbar_update >= 100:
            pbar.update(100)
            last_pbar_update = rounds_played
            last_update_time = time.time()
    
    # Final progress bar update
    if pbar is not None and rounds_played > last_pbar_update:
        pbar.update(rounds_played - last_pbar_update)
    
    return {
        'final_balance': info['balance'],
        'total_reward': total_reward,
        'rounds': rounds_played,
        'max_balance': max_balance,
        'min_balance': min_balance,
        'win_rate': info['win_rate'],
        'end_reason': 'max_rounds' if rounds_played >= max_rounds else info.get('reason', 'timeout' if time.time() - last_update_time > 5 else 'unknown')
    }

def pattern_betting_strategy(obs, last_results, current_streak, base_bet):
    """Bet based on patterns in last results."""
    if base_bet == 0:  # For conservative strategy
        return 0, 0
        
    # If we've played too many rounds without a pattern, quit
    if obs[0] > 50:  # If we've played more than 50 rounds
        return 6, 0  # Quit action
        
    # If we see 2 tails in a row, bet on heads
    if np.array_equal(last_results[-2:], np.array([1, 1])):  # Two tails
        return 1, base_bet  # Bet on heads
    # If we see 2 heads in a row, bet on tails
    elif np.array_equal(last_results[-2:], np.array([0, 0])):  # Two heads
        return 2, base_bet  # Bet on tails
    # If we see alternating results, bet on the opposite of the last result
    elif last_results[-1] != last_results[-2]:  # Alternating results
        return 2 if last_results[-1] == 0 else 1, base_bet // 2  # Bet half amount
    # Otherwise, no bet
    return 0, 0

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

def always_heads_strategy(obs, last_results, current_streak, base_bet):
    """Always bet on heads."""
    if base_bet == 0:  # For conservative strategy
        return 0, 0
    return 1, base_bet

def always_tails_strategy(obs, last_results, current_streak, base_bet):
    """Always bet on tails."""
    if base_bet == 0:  # For conservative strategy
        return 0, 0
    return 2, base_bet

def random_strategy(obs, last_results, current_streak, base_bet):
    """Random betting strategy."""
    action = np.random.randint(0, 3)  # 0: no bet, 1: heads, 2: tails
    if action == 0:  # If no bet, return 0 bet amount
        return action, 0
    # Randomly choose between base_bet and half base_bet
    bet_amount = base_bet if np.random.random() > 0.5 else max(5, base_bet // 2)
    return action, bet_amount

def conservative_strategy(obs, last_results, current_streak, base_bet):
    """Never bet - keep your money."""
    return 0, 0

def run_strategy_test(scenario_name: str, strategy_func, base_bet: int, num_episodes: int = 10000) -> Tuple[Dict, List[Dict]]:
    """Run multiple episodes for a single strategy."""
    env = TwoUpEnv(render_mode=None)
    episode_results = []
    end_reasons = defaultdict(int)
    
    for episode in range(num_episodes):
        result = run_episode(env, strategy_func, base_bet, None)
        episode_results.append(result)
        end_reasons[result['end_reason']] += 1
    
    # Calculate averages
    avg_balance = sum(r['final_balance'] for r in episode_results) / len(episode_results)
    avg_reward = sum(r['total_reward'] for r in episode_results) / len(episode_results)
    avg_rounds = sum(r['rounds'] for r in episode_results) / len(episode_results)
    avg_win_rate = sum(r['win_rate'] for r in episode_results) / len(episode_results)
    avg_max_balance = sum(r['max_balance'] for r in episode_results) / len(episode_results)
    
    strategy_result = [
        scenario_name,
        f"${avg_balance:.2f}",
        f"${avg_reward:.2f}",
        f"{avg_rounds:.1f}",
        f"{avg_win_rate:.1%}",
        f"${avg_max_balance:.2f}"
    ]
    
    return strategy_result, episode_results, end_reasons

def test_env(num_episodes: int = 10000):
    # Define test scenarios: (description, strategy_func, base_bet)
    scenarios = [
        ('Pattern Betting ($10 base)', pattern_betting_strategy, 10),
        ('Pattern Betting ($20 base)', pattern_betting_strategy, 20),
        ('Streak Betting ($10 base)', streak_betting_strategy, 10),
        ('Streak Betting ($20 base)', streak_betting_strategy, 20),
        ('Always Heads ($10 base)', always_heads_strategy, 10),
        ('Always Heads ($20 base)', always_heads_strategy, 20),
        ('Always Tails ($10 base)', always_tails_strategy, 10),
        ('Always Tails ($20 base)', always_tails_strategy, 20),
        ('Vibes Based ($10 base)', random_strategy, 10),
        ('Vibes Based ($20 base)', random_strategy, 20),
        ('Dickhead Strategy (No Bet)', conservative_strategy, 0),
    ]
    
    print("\nRunning Two Up Strategy Tests...")
    print("Note: Each episode has a maximum of 200 rounds")
    print("=" * 70)
    
    start_time = time.time()
    
    # Create a pool of workers
    num_cores = mp.cpu_count()
    print(f"Using {num_cores} CPU cores")
    print(f"Testing {len(scenarios)} strategies with {num_episodes} episodes each")
    print("=" * 70)
    
    # Track completed strategies
    completed_strategies = []
    results = []
    
    def callback(result):
        completed_strategies.append(1)
        results.append(result)
        print(f"Completed strategy {len(completed_strategies)}/{len(scenarios)}: {scenarios[len(completed_strategies)-1][0]}")
    
    # Run strategies in parallel
    with mp.Pool(processes=num_cores) as pool:
        # Submit all tasks
        for scenario in scenarios:
            pool.apply_async(run_strategy_test, (*scenario, num_episodes), callback=callback)
        
        # Wait for all tasks to complete
        pool.close()
        pool.join()
    
    # Process results
    strategy_results = []
    all_end_reasons = defaultdict(int)
    strategy_totals = defaultdict(lambda: {'episodes': 0, 'rounds': 0})
    
    for strategy_result, episode_results, end_reasons in results:
        strategy_results.append(strategy_result)
        
        # Update end reasons
        for reason, count in end_reasons.items():
            all_end_reasons[reason] += count
        
        # Update strategy totals
        scenario_name = strategy_result[0]
        strategy_totals[scenario_name]['episodes'] = len(episode_results)
        strategy_totals[scenario_name]['rounds'] = sum(r['rounds'] for r in episode_results)
    
    total_time = time.time() - start_time
    print(f"\nTotal testing time: {total_time:.2f} seconds")
    
    # Print strategy descriptions
    print("\nStrategy Descriptions:")
    print("=" * 70)
    for strategy_name, description in STRATEGY_DESCRIPTIONS.items():
        print(f"\n{strategy_name}:")
        print(description.strip())
    
    # Print totals for each strategy
    print("\nStrategy Totals:")
    print("=" * 70)
    for scenario_name, totals in strategy_totals.items():
        print(f"\n{scenario_name}:")
        print(f"  Total Episodes: {totals['episodes']}")
        print(f"  Total Rounds: {totals['rounds']}")
        print(f"  Average Rounds per Episode: {totals['rounds']/totals['episodes']:.1f}")
    
    # Print results table
    print("\nTwo Up Betting Strategy Analysis")
    print("=" * 70)
    print(f"Results averaged over {num_episodes} episodes per strategy")
    print("\nStrategy Results:")
    headers = ['Strategy', 'Avg Final Balance', 'Avg Total Reward', 'Avg Rounds', 'Avg Win Rate', 'Avg Max Balance']
    print(tabulate(strategy_results, headers=headers, tablefmt='grid'))
    
    # Print best performing strategies
    print("\nBest Performing Strategies:")
    print("1. Highest Average Balance:")
    best_balance = max(strategy_results, key=lambda x: float(x[1].replace('$', '')))
    print(f"   {best_balance[0]}: {best_balance[1]}")
    
    print("\n2. Highest Average Reward:")
    best_reward = max(strategy_results, key=lambda x: float(x[2].replace('$', '')))
    print(f"   {best_reward[0]}: {best_reward[2]}")
    
    print("\n3. Most Rounds Played:")
    most_rounds = max(strategy_results, key=lambda x: float(x[3]))
    print(f"   {most_rounds[0]}: {most_rounds[3]} rounds")
    
    print("\n4. Highest Win Rate:")
    best_win_rate = max(strategy_results, key=lambda x: float(x[4].strip('%'))/100)
    print(f"   {best_win_rate[0]}: {best_win_rate[4]}")
    
    # Print episode end reasons
    print("\nEpisode End Reasons:")
    for reason, count in all_end_reasons.items():
        print(f"   {reason}: {count} episodes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Two Up strategy tests')
    parser.add_argument('--episodes', type=int, default=10000,
                      help='Number of episodes to run per strategy (default: 10000)')
    args = parser.parse_args()
    
    test_env(num_episodes=args.episodes) 