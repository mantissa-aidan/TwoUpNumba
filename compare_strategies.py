from stable_baselines3 import PPO
from two_up_env import TwoUpEnv
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def run_simulation(env, model=None, num_episodes=1000):
    """Run multiple episodes and collect statistics"""
    results = defaultdict(list)
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            if model is None:
                # Random strategy
                action = np.random.randint(0, 3)  # 0: no bet, 1: heads, 2: tails
            else:
                action, _ = model.predict(obs)
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            if done:
                results['final_balance'].append(info['balance'])
                results['win_rate'].append(info['win_rate'])
                results['episode_reward'].append(episode_reward)
    
    return results

def plot_results(random_results, trained_results, no_bet_results):
    """Plot comparison of different strategies"""
    plt.figure(figsize=(15, 5))
    
    # Plot final balances
    plt.subplot(1, 3, 1)
    plt.boxplot([random_results['final_balance'], 
                 trained_results['final_balance'],
                 no_bet_results['final_balance']],
                labels=['Random', 'Trained', 'No Bet'])
    plt.title('Final Balance Distribution')
    plt.ylabel('Balance ($)')
    
    # Plot win rates
    plt.subplot(1, 3, 2)
    plt.boxplot([random_results['win_rate'], 
                 trained_results['win_rate'],
                 no_bet_results['win_rate']],
                labels=['Random', 'Trained', 'No Bet'])
    plt.title('Win Rate Distribution')
    plt.ylabel('Win Rate')
    
    # Plot episode rewards
    plt.subplot(1, 3, 3)
    plt.boxplot([random_results['episode_reward'], 
                 trained_results['episode_reward'],
                 no_bet_results['episode_reward']],
                labels=['Random', 'Trained', 'No Bet'])
    plt.title('Episode Reward Distribution')
    plt.ylabel('Total Reward')
    
    plt.tight_layout()
    plt.savefig('strategy_comparison.png')
    plt.close()

def main():
    # Create environments
    env_random = TwoUpEnv()
    env_trained = TwoUpEnv()
    env_no_bet = TwoUpEnv()
    
    # Train an agent
    print("Training agent...")
    model = PPO("MlpPolicy", env_trained, verbose=1)
    model.learn(total_timesteps=100000)
    
    # Run simulations
    print("\nRunning simulations...")
    num_episodes = 1000
    
    print("Running random strategy...")
    random_results = run_simulation(env_random, num_episodes=num_episodes)
    
    print("Running trained agent...")
    trained_results = run_simulation(env_trained, model=model, num_episodes=num_episodes)
    
    print("Running no-bet strategy...")
    no_bet_results = run_simulation(env_no_bet, num_episodes=num_episodes)
    
    # Print statistics
    print("\nResults Summary:")
    print("\nRandom Strategy:")
    print(f"Average Final Balance: ${np.mean(random_results['final_balance']):.2f}")
    print(f"Average Win Rate: {np.mean(random_results['win_rate']):.2%}")
    
    print("\nTrained Agent:")
    print(f"Average Final Balance: ${np.mean(trained_results['final_balance']):.2f}")
    print(f"Average Win Rate: {np.mean(trained_results['win_rate']):.2%}")
    
    print("\nNo-Bet Strategy:")
    print(f"Average Final Balance: ${np.mean(no_bet_results['final_balance']):.2f}")
    print(f"Average Win Rate: {np.mean(no_bet_results['win_rate']):.2%}")
    
    # Plot results
    plot_results(random_results, trained_results, no_bet_results)
    print("\nResults have been plotted to 'strategy_comparison.png'")

if __name__ == "__main__":
    main() 