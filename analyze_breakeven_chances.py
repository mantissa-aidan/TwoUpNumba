import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any

def find_latest_json_report(report_dir: str, report_prefix: str, personality_suffix: Optional[str] = None) -> Optional[str]:
    """Find the most recent JSON report file matching the pattern."""
    if personality_suffix:
        search_pattern = os.path.join(report_dir, f'{report_prefix}_{personality_suffix}_*.json')
    else:
        search_pattern = os.path.join(report_dir, f'{report_prefix}_*.json')
    
    search_pattern = search_pattern.replace('\\', '/') # Ensure forward slashes for glob
    report_files = glob.glob(search_pattern)
    
    if not report_files:
        print(f"No files found for pattern: {search_pattern}")
        return None
    return max(report_files, key=os.path.getmtime)

def load_json_data(file_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load data from a JSON report file."""
    if not file_path or not os.path.exists(file_path):
        print(f'Warning: Report file not found or path is None: {file_path}')
        return None
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(f'Error decoding JSON from {file_path}: {e}')
        return None
    except Exception as e:
        print(f'Error loading JSON report {file_path}: {e}')
        return None

def calculate_breakeven_probability(final_balances: List[float], initial_balance: float) -> float:
    """Calculate the percentage of episodes where final balance >= initial balance."""
    if not final_balances: # Handle empty list
        return 0.0
    breakeven_or_better_count = sum(1 for balance in final_balances if balance >= initial_balance)
    return (breakeven_or_better_count / len(final_balances)) * 100

def plot_breakeven_probabilities(strategy_probabilities: List[Tuple[str, float]], output_dir: str = "."):
    """Plot the break-even probabilities as a bar chart."""
    if not strategy_probabilities:
        print("No data to plot.")
        return

    # Sort by probability for better visualization
    strategy_probabilities.sort(key=lambda x: x[1], reverse=True)
    
    names = [item[0] for item in strategy_probabilities]
    probabilities = [item[1] for item in strategy_probabilities]

    plt.figure(figsize=(max(10, len(names) * 0.5), 8)) # Dynamic width
    bars = plt.bar(names, probabilities, color='skyblue')
    plt.ylabel('Probability of Breaking Even or Better (%)')
    plt.xlabel('Strategy')
    plt.title('Chance of Breaking Even (Final Balance >= Initial Balance)')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(np.arange(0, 101, 10))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Add text labels on bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=8)

    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, "breakeven_chance_comparison.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Break-even probability plot saved to: {plot_filename}")

def main():
    parser = argparse.ArgumentParser(description='Analyze strategies for their chance of breaking even or doing better.')
    parser.add_argument('--initial_balance', type=float, default=100.0, help='Initial balance to consider for break-even calculation.')
    parser.add_argument('--plot', action='store_true', help='Generate a plot of the results.')
    args = parser.parse_args()

    all_strategy_breakeven_chances: List[Tuple[str, float]] = []

    # 1. Load Universal Rule-Based Strategies Report
    print("--- Loading Universal Rule-Based Strategy Data ---")
    rule_based_report_dir = "reports"
    rule_based_prefix = "rule_based_strategies_summary_universal"
    latest_rule_report_path = find_latest_json_report(rule_based_report_dir, rule_based_prefix)
    
    if latest_rule_report_path:
        rule_data = load_json_data(latest_rule_report_path)
        if rule_data and "strategies" in rule_data:
            print(f"Loaded rule-based data from: {latest_rule_report_path}")
            for strategy_summary in rule_data["strategies"]:
                name = strategy_summary.get("strategy_name", "Unknown Rule-Based")
                raw_balances = strategy_summary.get("final_balance_raw_data")
                if raw_balances is not None:
                    prob = calculate_breakeven_probability(raw_balances, args.initial_balance)
                    all_strategy_breakeven_chances.append((f"{name} (Rule-Based)", prob))
                else:
                    print(f"  Warning: Raw final balance data not found for rule-based strategy: {name}")
        else:
            print("Could not parse strategies from the loaded rule-based report.")
    else:
        print("Universal rule-based strategies report not found.")
    print("Completed rule-based analysis.\n")

    # 2. Load PPO Agent Reports (for each personality)
    print("--- Loading PPO Agent Data (All Personalities) ---")
    personalities = ["standard", "thrill_seeker", "loss_averse", "play_for_time"]
    ppo_report_prefix = "ppo_vs_baselines"

    for personality in personalities:
        print(f"Processing PPO data for personality: '{personality}'...")
        ppo_report_dir = os.path.join("personality_agents", personality, "reports")
        latest_ppo_report_path = find_latest_json_report(ppo_report_dir, ppo_report_prefix, personality_suffix=personality)
        
        if latest_ppo_report_path:
            ppo_data = load_json_data(latest_ppo_report_path)
            if ppo_data and "strategy_results" in ppo_data:
                print(f"  Loaded PPO data from: {latest_ppo_report_path}")
                for ppo_strategy_key, ppo_summary in ppo_data["strategy_results"].items():
                    # Construct a unique name, e.g., "PPO Trained (standard)", "Random Baseline (standard)"
                    base_name = ppo_strategy_key.replace("_", " ").title()
                    if "Ppo" in base_name: base_name = base_name.replace("Ppo", "PPO") # Correct casing
                    full_strategy_name = f"{base_name} ({personality})"
                    
                    raw_balances = ppo_summary.get("final_balance_raw_data")
                    if raw_balances is not None:
                        prob = calculate_breakeven_probability(raw_balances, args.initial_balance)
                        all_strategy_breakeven_chances.append((full_strategy_name, prob))
                    else:
                        print(f"    Warning: Raw final balance data not found for PPO strategy: {full_strategy_name}")
            else:
                print(f"  Could not parse strategies from the loaded PPO report for '{personality}'.")
        else:
            print(f"  PPO comparison report not found for personality: '{personality}'.")
    print("Completed PPO agent analysis.\n")

    # 3. Display Results
    print("--- Break-Even or Better Probabilities ---")
    if not all_strategy_breakeven_chances:
        print("No strategy data loaded to analyze.")
        return

    # Sort by probability, descending
    all_strategy_breakeven_chances.sort(key=lambda x: x[1], reverse=True)

    print(f"(Initial Balance for Break-Even: ${args.initial_balance:.2f})")
    for name, prob in all_strategy_breakeven_chances:
        print(f"  {name:<50}: {prob:.2f}%")
    
    # 4. Optional Plotting
    if args.plot:
        print("\nGenerating plot...")
        # Save plot in a general 'reports' directory or a specific analysis directory
        plot_output_dir = "reports/analysis_results"
        plot_breakeven_probabilities(all_strategy_breakeven_chances, output_dir=plot_output_dir)

if __name__ == "__main__":
    main() 