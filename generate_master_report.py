import os
import json
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
from typing import Dict, Optional, Any, List

def find_latest_report(report_dir: str, report_prefix: str, personality: Optional[str] = None) -> Optional[str]:
    """Find the most recent report file matching the pattern.
       If personality is None, it's omitted from the pattern.
    """
    if personality:
        search_pattern = os.path.join(report_dir, f"{report_prefix}_{personality}_*.json")
    else:
        search_pattern = os.path.join(report_dir, f"{report_prefix}_*.json")
    
    search_pattern = search_pattern.replace('\\', '/')
    report_files = glob.glob(search_pattern)
    if not report_files:
        print(f"No files found for pattern: {search_pattern}")
        return None
    return max(report_files, key=os.path.getmtime)

def load_json_report(file_path: Optional[str]) -> Optional[Any]: # Return type can be List or Dict
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

def plot_combined_results(all_strategy_plot_data: Dict[str, Dict[str, List[Any]]], 
                          personality_context: str, output_dir: str, num_episodes: int):
    """Plot comparison of all strategies based on their average final balances and std dev."""
    strategy_names = list(all_strategy_plot_data.keys())
    avg_balances = [all_strategy_plot_data[name]['avg_final_balance'][0] for name in strategy_names]
    std_devs = [all_strategy_plot_data[name]['std_dev_final_balance'][0] for name in strategy_names]

    if not strategy_names:
        print("No data to plot for master report.")
        return

    plt.figure(figsize=(max(10, len(strategy_names) * 1.5), 7))
    bars = plt.bar(strategy_names, avg_balances, yerr=std_devs, capsize=5, color='skyblue')
    plt.ylabel(f'Average Final Balance (after {num_episodes} episodes)')
    plt.title(f'Master Strategy Comparison (PPO: {personality_context})\nAvg Final Balance & StdDev')
    plt.xticks(rotation=30, ha="right") # Adjusted rotation
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + np.sign(yval)*0.02*max(avg_balances, default=0), 
                 f'{yval:.2f}', ha='center', va='bottom' if yval >=0 else 'top')

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f'master_summary_plot_{personality_context}_{timestamp_str}.png')
    try:
        plt.savefig(plot_filename)
        plt.close()
        print(f"Master comparison plot saved to: {plot_filename}")
    except Exception as e:
        print(f"Error saving master plot {plot_filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Generate a master report by combining JSON summaries.')
    parser.add_argument('--personality', type=str, required=True, 
                        choices=['standard', 'thrill_seeker', 'loss_averse', 'play_for_time'], 
                        help='PPO agent personality to include in the report.')
    # Add env params for documentation/consistency, though not directly used for env creation in this script
    parser.add_argument("--initial-balance", type=float, default=100.0, help="Initial balance (contextual).")
    parser.add_argument("--base-bet", type=float, default=10.0, help="Base bet unit (contextual).")
    parser.add_argument("--max-episode-steps", type=int, default=200, help="Max steps per episode (contextual).")
    parser.add_argument("--num-episodes-context", type=int, default=1000, help="Number of episodes run for component reports (contextual).")

    args = parser.parse_args()

    ppo_report_dir = os.path.join("personality_agents", args.personality, "reports")
    ppo_report_prefix = "ppo_vs_baselines"
    ppo_report_path = find_latest_report(ppo_report_dir, ppo_report_prefix, args.personality)
    ppo_data_list = load_json_report(ppo_report_path) # This is a LIST of strategy summaries

    rule_based_report_dir = "reports"
    rule_based_prefix = "rule_based_strategies_summary_universal"
    rule_based_report_path = find_latest_report(rule_based_report_dir, rule_based_prefix) # No personality for universal
    rule_based_data_list = load_json_report(rule_based_report_path) # This is also a LIST of strategy summaries

    if not ppo_data_list and not rule_based_data_list:
        print("No PPO or rule-based data found. Exiting.")
        return

    # Stores strategy_name -> {metric_name: [value], ...}, focusing on avg and std_dev for plotting
    plot_data_for_master: Dict[str, Dict[str, List[Any]]] = defaultdict(lambda: defaultdict(list))
    combined_json_summaries: Dict[str, Any] = {}

    # Process PPO vs Baselines data (list of dicts)
    if ppo_data_list and isinstance(ppo_data_list, list):
        print(f"Processing PPO report: {os.path.abspath(ppo_report_path) if ppo_report_path else 'N/A'}")
        for summary_item in ppo_data_list:
            name = summary_item.get("strategy_name", "Unknown PPO/Baseline Strategy")
            # Add (Personality) to PPO agent name for clarity in combined report
            if name == "PPO Agent": name = f"PPO Agent ({args.personality})"
            
            combined_json_summaries[name] = summary_item 
            if summary_item.get('avg_final_balance') is not None:
                plot_data_for_master[name]['avg_final_balance'].append(summary_item['avg_final_balance'])
            if summary_item.get('std_dev_final_balance') is not None:
                plot_data_for_master[name]['std_dev_final_balance'].append(summary_item['std_dev_final_balance'])
            # Add other metrics if needed for the plot later, like avg_episode_reward, win_rate etc.
            # For this plot, we primarily use avg_final_balance and its std_dev.
    else:
        print(f"Warning: PPO data for '{args.personality}' not found or in unexpected format.")

    # Process Universal Rule-Based data (list of dicts)
    if rule_based_data_list and isinstance(rule_based_data_list, list):
        print(f"Processing Universal Rule-Based report: {os.path.abspath(rule_based_report_path) if rule_based_report_path else 'N/A'}")
        for strategy_summary_item in rule_based_data_list:
            strategy_name = strategy_summary_item.get("strategy_name", "Unknown Rule-Based Strategy")
            unique_name = strategy_name
            counter = 1
            while unique_name in combined_json_summaries: # Avoid name collision
                unique_name = f"{strategy_name} ({counter})"
                counter += 1
            
            combined_json_summaries[unique_name] = strategy_summary_item
            if strategy_summary_item.get('avg_final_balance') is not None:
                plot_data_for_master[unique_name]['avg_final_balance'].append(strategy_summary_item['avg_final_balance'])
            if strategy_summary_item.get('std_dev_final_balance') is not None:
                plot_data_for_master[unique_name]['std_dev_final_balance'].append(strategy_summary_item['std_dev_final_balance'])
    else:
        print(f"Warning: Universal rule-based data not found or in unexpected format.")

    # Create the final combined JSON structure
    final_master_json_content: Dict[str, Any] = {
        "report_timestamp": datetime.datetime.now().isoformat(),
        "combined_report_for_personality_context": args.personality,
        "environment_parameters_context": {
            "initial_balance": args.initial_balance,
            "base_bet_unit": args.base_bet,
            "max_episode_steps": args.max_episode_steps,
            "num_episodes_per_strategy_run": args.num_episodes_context
        },
        "source_reports": {
            "ppo_vs_baselines": os.path.abspath(ppo_report_path) if ppo_report_path and ppo_data_list else None,
            "universal_rule_based_strategies": os.path.abspath(rule_based_report_path) if rule_based_report_path and rule_based_data_list else None
        },
        "strategy_summaries": combined_json_summaries
    }

    if not final_master_json_content["strategy_summaries"]:
        print("No strategy summary data was successfully combined for the JSON report.")
    else:
        master_report_save_dir = os.path.join("personality_agents", args.personality, "reports")
        os.makedirs(master_report_save_dir, exist_ok=True)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        master_report_filename = os.path.join(master_report_save_dir, f'master_summary_report_{args.personality}_{timestamp_str}.json')
        try:
            with open(master_report_filename, 'w') as f:
                json.dump(final_master_json_content, f, indent=4)
            print(f"Master JSON summary report saved to: {master_report_filename}")
        except Exception as e:
            print(f"Error saving master JSON report {master_report_filename}: {e}")

    if plot_data_for_master:
        print("\nGenerating master plot (Avg Final Balance & StdDev)...")
        plot_combined_results(plot_data_for_master, args.personality, master_report_save_dir, args.num_episodes_context)
    else:
        print("\nNo data prepared to generate master plot.")

if __name__ == "__main__":
    main() 