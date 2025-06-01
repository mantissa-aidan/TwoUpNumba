import os
import json
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
from typing import Dict, Optional, Any, List

def find_latest_report(report_dir: str, report_prefix: str, personality: str) -> Optional[str]:
    """Find the most recent report file matching the pattern."""
    # Using f-string directly for simpler path construction
    search_pattern = f"{report_dir}/{report_prefix}_{personality}_*.json"
    # Replace backslashes with forward slashes for consistency, glob handles it well
    search_pattern = search_pattern.replace('\\', '/')
    report_files = glob.glob(search_pattern)
    if not report_files:
        print(f"No files found for pattern: {search_pattern}")
        return None
    return max(report_files, key=os.path.getmtime)

def load_json_report(file_path: Optional[str]) -> Optional[Dict[str, Any]]:
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

def plot_combined_results(all_strategy_plot_data: Dict[str, Dict[str, List[Any]]], personality: str, output_dir: str):
    """Plot comparison of all strategies. Expects dict where values are dicts of metric lists (e.g., {'metric_name': [value1, value2]})"""
    num_strategies = len(all_strategy_plot_data)
    if num_strategies == 0:
        print("No combined strategy results to plot.")
        return

    plt.figure(figsize=(min(max(12, num_strategies * 2.5), 30), 18))
    strategy_names = list(all_strategy_plot_data.keys())
    
    metrics_to_plot = [
        ('final_balance', f'Final Balance Distribution (from summaries - {personality})', 'Balance ($)'),
        ('win_rate', f'Win Rate Distribution (from summaries - {personality})', 'Win Rate'),
        ('episode_reward', f'Episode Reward Distribution (from summaries - {personality})', 'Total Reward')
    ]

    for i, (metric_key, title, ylabel) in enumerate(metrics_to_plot):
        plt.subplot(len(metrics_to_plot), 1, i + 1)
        
        plot_data_for_metric: List[List[Any]] = []
        valid_labels_for_metric: List[str] = []
        
        for name in strategy_names:
            if metric_key in all_strategy_plot_data[name] and \
               isinstance(all_strategy_plot_data[name][metric_key], list) and \
               all_strategy_plot_data[name][metric_key] and \
               all_strategy_plot_data[name][metric_key][0] is not None:
                plot_data_for_metric.append(all_strategy_plot_data[name][metric_key])
                valid_labels_for_metric.append(name)
            else:
                print(f"Info: Metric '{metric_key}' data not available or not suitable for strategy '{name}'. Skipping in subplot.")
        
        if not plot_data_for_metric or not valid_labels_for_metric:
            print(f"Warning: No valid data to plot for metric '{metric_key}' across all strategies. Skipping subplot.")
            continue
            
        try:
            # Boxplot expects a list of lists if multiple series are plotted with multiple labels
            # If each strategy provides a list (even of one item, like an average), this works.
            plt.boxplot(plot_data_for_metric, labels=valid_labels_for_metric, vert=False, patch_artist=True, medianprops=dict(color="red", linewidth=1.5))
        except Exception as e:
            print(f"Error during boxplot for {metric_key}: {e}. Data for metric: {plot_data_for_metric}")
            continue

        plt.title(title)
        plt.xlabel(ylabel)
        plt.yticks(fontsize=9)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'Comprehensive Strategy Comparison for {personality} Personality (from Summaries)\nReport Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', fontsize=14)
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f'master_summary_plot_{personality}_{timestamp_str}.png')
    try:
        plt.savefig(plot_filename)
        plt.close() # Close the figure to free memory
        print(f"Master comparison plot (from summaries) saved to: {plot_filename}")
    except Exception as e:
        print(f"Error saving plot {plot_filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Generate a master report by combining JSON summaries from PPO and rule-based strategy evaluations.')
    parser.add_argument('--personality', type=str, required=True, choices=['standard', 'thrill_seeker', 'loss_averse', 'play_for_time'], help='Gambler personality of the PPO agent to generate the report for.')
    args = parser.parse_args()

    # PPO reports are personality-specific
    ppo_report_dir = os.path.join("personality_agents", args.personality, "reports")
    abs_ppo_report_dir = os.path.abspath(ppo_report_dir)
    print(f"Looking for PPO reports in: {abs_ppo_report_dir} for personality '{args.personality}'")

    ppo_report_file_prefix = "ppo_vs_baselines"
    ppo_report_path = find_latest_report(ppo_report_dir, ppo_report_file_prefix, args.personality)
    ppo_data = load_json_report(ppo_report_path)
    if ppo_data:
        print(f"Loaded PPO report: {os.path.abspath(ppo_report_path) if ppo_report_path else 'N/A'}")
    else:
        print(f"Could not load PPO report for personality '{args.personality}'. Searched with prefix '{ppo_report_file_prefix}' in {ppo_report_dir}")

    # Rule-based reports are now universal
    universal_rule_based_report_dir = "reports" # General reports directory
    abs_universal_rule_based_report_dir = os.path.abspath(universal_rule_based_report_dir)
    rule_based_report_file_prefix = "rule_based_strategies_summary_universal"
    # The find_latest_report function needs adjustment if personality is not part of prefix or make a more generic find_latest
    # For now, let's assume find_latest_report can work if we pass a dummy/empty personality or adjust it.
    # Let's make a small helper or modify find_latest_report to handle no personality in prefix.
    
    # Simplified search for universal rule-based report (assumes find_latest_report can handle it or we make a specific one)
    # search_pattern_universal = os.path.join(universal_rule_based_report_dir, f"{rule_based_report_file_prefix}_*.json")
    # report_files_universal = glob.glob(search_pattern_universal.replace('\\', '/'))
    # rule_based_report_path = max(report_files_universal, key=os.path.getmtime) if report_files_universal else None
    
    # Using a modified approach to find the latest universal report, assuming only one such file exists or we want the newest if multiple somehow created.
    # To reuse find_latest_report, we can pass an empty string or specific marker if it expects personality in pattern.
    # Let's assume for now we modify find_latest_report to be more flexible or create a new one.
    # For this edit, let's assume find_latest_report is adapted or we handle it simply:
    rule_based_reports_search_path = os.path.join(universal_rule_based_report_dir, f"{rule_based_report_file_prefix}_*.json")
    rule_based_report_files = sorted(glob.glob(rule_based_reports_search_path.replace('\\','/')), key=os.path.getmtime, reverse=True)
    rule_based_report_path = rule_based_report_files[0] if rule_based_report_files else None

    rule_based_data = load_json_report(rule_based_report_path)

    if rule_based_data:
        print(f"Loaded UNIVERSAL Rule-Based report: {os.path.abspath(rule_based_report_path) if rule_based_report_path else 'N/A'}")
    else:
        print(f"Could not load UNIVERSAL Rule-Based report. Searched in '{universal_rule_based_report_dir}' with prefix '{rule_based_report_file_prefix}'")

    if not ppo_data and not rule_based_data: # Check if at least one data source is available
        print(f"No PPO data for '{args.personality}' AND no universal rule-based data found. Exiting.")
        return
    if not ppo_data:
        print(f"Warning: PPO data for '{args.personality}' not found. Report will only contain rule-based strategies.")
    if not rule_based_data:
        print(f"Warning: Universal rule-based data not found. Report will only contain PPO agent for '{args.personality}'.")

    print("\nNOTE: This script combines PPO agent summary statistics (for a specific personality) with universal rule-based strategy summary statistics.")
    print("The 'Average Episode Reward' for rule-based strategies reflects rewards from a 'standard' environment context.")
    print("The primary cross-comparison metric is 'Average Final Balance'.\n")

    # This will store strategy_name -> {metric_name: [value_from_summary], ...}
    # For boxplot, each metric should be a list of values. Since we have summaries, it will be a list with one item (the average).
    plot_data_for_master: Dict[str, Dict[str, List[Any]]] = defaultdict(lambda: defaultdict(list))

    combined_json_content: Dict[str, Any] = {
        "report_timestamp": datetime.datetime.now().isoformat(),
        "combined_report_for_personality": args.personality,
        "source_reports": {
            "ppo_vs_baselines": os.path.abspath(ppo_report_path) if ppo_report_path else None,
            "universal_rule_based_strategies": os.path.abspath(rule_based_report_path) if rule_based_report_path else None
        },
        "strategy_summaries": {} # This will hold the actual summary data
    }

    if ppo_data and "strategy_results" in ppo_data:
        for strategy_key, summary_stats in ppo_data["strategy_results"].items():
            name = strategy_key.replace("_", " ").title()
            if "Ppo" in name: name = name.replace("Ppo", "PPO") # Correct PPO casing
            
            combined_json_content["strategy_summaries"][name] = summary_stats
            
            # Prepare for plotting: wrap averages in a list for boxplot
            if summary_stats.get('average_final_balance') is not None:
                plot_data_for_master[name]['final_balance'].append(summary_stats['average_final_balance'])
            if summary_stats.get('average_win_rate') is not None:
                plot_data_for_master[name]['win_rate'].append(summary_stats['average_win_rate'])
            if summary_stats.get('average_episode_reward') is not None:
                plot_data_for_master[name]['episode_reward'].append(summary_stats['average_episode_reward'])

    if rule_based_data and "strategies" in rule_based_data: # 'strategies' is a list of summaries
        for strategy_summary_item in rule_based_data["strategies"]:
            strategy_name = strategy_summary_item.get("strategy_name", "Unknown Rule-Based Strategy")
            
            # Basic disambiguation if a rule-based strategy has a clashing name (unlikely with current setup)
            unique_name = strategy_name
            counter = 1
            while unique_name in combined_json_content["strategy_summaries"]:
                unique_name = f"{strategy_name} ({counter})"
                counter += 1
            
            combined_json_content["strategy_summaries"][unique_name] = strategy_summary_item
            
            # Prepare for plotting: wrap averages in a list
            if strategy_summary_item.get('avg_final_balance') is not None:
                plot_data_for_master[unique_name]['final_balance'].append(strategy_summary_item['avg_final_balance'])
            if strategy_summary_item.get('avg_win_rate') is not None:
                plot_data_for_master[unique_name]['win_rate'].append(strategy_summary_item['avg_win_rate'])
            if strategy_summary_item.get('avg_episode_reward') is not None:
                plot_data_for_master[unique_name]['episode_reward'].append(strategy_summary_item['avg_episode_reward'])

    if not combined_json_content["strategy_summaries"]:
        print("No strategy summary data was successfully combined for the JSON report.")
    else:
        os.makedirs(ppo_report_dir, exist_ok=True)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        master_report_filename = os.path.join(ppo_report_dir, f'master_summary_report_{args.personality}_{timestamp_str}.json')
        try:
            with open(master_report_filename, 'w') as f:
                json.dump(combined_json_content, f, indent=4)
            print(f"Master JSON summary report saved to: {master_report_filename}")
        except Exception as e:
            print(f"Error saving master JSON report {master_report_filename}: {e}")

    if plot_data_for_master:
        print("\nGenerating master plot...")
        # The plot function will show PPO (personality) vs Universal Rule-Based (standard rewards for avg_ep_rew)
        plot_combined_results(plot_data_for_master, args.personality, ppo_report_dir) # Save plot in PPO personality's report dir
    else:
        print("\nNo data prepared to generate master plot. This might be due to missing source files or empty summaries.")

if __name__ == "__main__":
    main() 