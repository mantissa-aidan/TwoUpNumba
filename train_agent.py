from stable_baselines3 import PPO
from two_up_env import TwoUpEnv
import numpy as np

def main():
    # Create the environment
    env = TwoUpEnv()
    
    # Create the agent (PPO algorithm)
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train the agent
    print("Training the agent...")
    model.learn(total_timesteps=100000)
    
    # Test the trained agent
    print("\nTesting the trained agent...")
    obs = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
    
    print(f"\nFinal balance: ${info['balance']}")
    print(f"Total reward: {total_reward}")

if __name__ == "__main__":
    main() 