import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Any

REPORTS_DIR = "reports"
PERSONALITY_AGENTS_DIR = "personality_agents"

def find_latest_report_file(directory: str, prefix: str, personality: Optional[str] = None) -> Optional[str]:
    """Finds the latest report file matching a prefix in a directory."""
    if personality:
        pattern = os.path.join(directory, f"{prefix}_{personality}_*.json")
    else:
        pattern = os.path.join(directory, f"{prefix}_*.json")
    
    files = glob.glob(pattern.replace('\\', '/'))
    if not files:
        print(f"No files found for pattern: {pattern}")
        return None
    return max(files, key=os.path.getmtime)

def load_json_data(file_path: Optional[str]) -> Optional[Any]:
    """Loads JSON data from a file path."""
    if not file_path or not os.path.exists(file_path):
        print(f"Warning: File not found or path is None: {file_path}")
        return None
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {file_path}: {e}")
        return None

def calculate_breakeven_probability(raw_final_balances: List[float], initial_balance: float) -> float:
    """Calculates the probability of final balance >= initial balance."""
    if not raw_final_balances:
        return 0.0
    breakeven_or_better_count = sum(1 for balance in raw_final_balances if balance >= initial_balance)
    return (breakeven_or_better_count / len(raw_final_balances)) * 100

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
    parser = argparse.ArgumentParser(description="Analyze break-even chances from PPO and rule-based strategy reports.")
    parser.add_argument("--initial-balance", type=float, default=100.0, 
                        help="Initial balance to consider for break-even analysis (must match training/eval context). Default: 100.0")
    parser.add_argument("--plot", action='store_true', help="Generate a bar chart of break-even probabilities.")
    args = parser.parse_args()

    all_strategy_breakeven_chances: List[Tuple[str, float]] = []

    # 1. Load Universal Rule-Based Strategies Report
    rule_based_report_path = find_latest_report_file(REPORTS_DIR, "rule_based_strategies_summary_universal")
    rule_based_data_list = load_json_data(rule_based_report_path)

    if rule_based_data_list and isinstance(rule_based_data_list, list):
        print(f"Processing universal rule-based report: {rule_based_report_path}")
        for strategy_summary in rule_based_data_list:
            name = strategy_summary.get("strategy_name", "Unknown Rule-Based Strategy")
            raw_data = strategy_summary.get("final_balance_raw_data")
            if raw_data:
                prob = calculate_breakeven_probability(raw_data, args.initial_balance)
                all_strategy_breakeven_chances.append((name, prob))
            else:
                print(f"Warning: No 'final_balance_raw_data' for {name} in rule-based report.")
    else:
        print(f"Warning: Universal rule-based report not found or in unexpected format at {rule_based_report_path}")

    # 2. Load PPO vs. Baselines reports for all personalities
    if not os.path.isdir(PERSONALITY_AGENTS_DIR):
        print(f"Directory {PERSONALITY_AGENTS_DIR} not found. Skipping PPO agent reports.")
    else:
        personalities = [d for d in os.listdir(PERSONALITY_AGENTS_DIR) if os.path.isdir(os.path.join(PERSONALITY_AGENTS_DIR, d))]
        print(f"Found personalities: {personalities}")

        for personality_name in personalities:
            ppo_personality_report_dir = os.path.join(PERSONALITY_AGENTS_DIR, personality_name, "reports")
            ppo_report_path = find_latest_report_file(ppo_personality_report_dir, "ppo_vs_baselines", personality_name)
            ppo_data_list = load_json_data(ppo_report_path) # This is a LIST of summaries

            if ppo_data_list and isinstance(ppo_data_list, list):
                print(f"Processing PPO report for {personality_name}: {ppo_report_path}")
                for strategy_summary in ppo_data_list:
                    name = strategy_summary.get("strategy_name", f"Unknown Strategy from {personality_name}")
                    # Augment PPO agent name with its personality for clarity
                    if name == "PPO Agent": 
                        name = f"PPO Agent ({personality_name})"
                    # Also augment Random and NoBet if they appear here, to distinguish from universal ones if names clash
                    elif name == "Random (Baseline)":
                         name = f"Random (Baseline - {personality_name} context)"
                    elif name == "Always No Bet":
                         name = f"Always No Bet ({personality_name} context)"

                    raw_data = strategy_summary.get("final_balance_raw_data")
                    if raw_data:
                        prob = calculate_breakeven_probability(raw_data, args.initial_balance)
                        # Avoid adding duplicate strategy names if somehow present from different files (shouldn't happen with this naming)
                        existing_names = [s[0] for s in all_strategy_breakeven_chances]
                        if name not in existing_names:
                            all_strategy_breakeven_chances.append((name, prob))
                        else:
                            print(f"Warning: Strategy '{name}' already processed. Skipping duplicate entry from {personality_name} report.")
                    else:
                        print(f"Warning: No 'final_balance_raw_data' for {name} in {personality_name} PPO report.")
            else:
                print(f"Warning: PPO report for {personality_name} not found or in unexpected format at {ppo_report_path}")

    if not all_strategy_breakeven_chances:
        print("No strategy data found to analyze break-even chances.")
        return

    # Sort by probability (descending)
    all_strategy_breakeven_chances.sort(key=lambda x: x[1], reverse=True)

    print("\n--- Break-Even or Better Probabilities ---")
    print(f"(Based on initial balance of: ${args.initial_balance:.2f})")
    for name, prob in all_strategy_breakeven_chances:
        print(f"- {name}: {prob:.2f}%")

    if args.plot:
        print("\nGenerating plot...")
        names = [s[0] for s in all_strategy_breakeven_chances]
        probs = [s[1] for s in all_strategy_breakeven_chances]
        
        plt.figure(figsize=(max(10, len(names) * 0.6), 8))
        bars = plt.barh(names, probs, color='teal') # Horizontal bar chart for better readability
        plt.xlabel('Probability of Final Balance >= Initial Balance (%)')
        plt.title(f'Break-Even or Better Probability (Initial Balance: ${args.initial_balance:.2f})')
        plt.gca().invert_yaxis() # Display highest probability at the top
        plt.tight_layout()

        for bar in bars:
            plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                     f'{bar.get_width():.2f}%', va='center')

        analysis_results_dir = os.path.join(REPORTS_DIR, "analysis_results")
        os.makedirs(analysis_results_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plot_filename = os.path.join(analysis_results_dir, f"breakeven_chances_{timestamp}.png")
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
        # plt.show()

if __name__ == "__main__":
    main() 