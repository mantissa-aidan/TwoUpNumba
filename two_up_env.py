import gym
import numpy as np
from gym import spaces
import random

class TwoUpEnv(gym.Env):
    """
    Two Up Environment for Reinforcement Learning
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(TwoUpEnv, self).__init__()
        
        # Define action space (0: no bet, 1: bet on heads, 2: bet on tails)
        self.action_space = spaces.Discrete(3)
        
        # Define observation space
        # [current_streak, last_3_results, current_bet_amount, balance]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0]),
            high=np.array([10, 1, 1, 1, 1, 100, 1000]),
            dtype=np.float32
        )
        
        # Game state
        self.current_streak = 0
        self.last_3_results = [0, 0, 0]  # 0: heads, 1: tails, 2: odds
        self.current_bet = 0
        self.balance = 100  # Starting balance
        self.consecutive_odds = 0
        self.total_games = 0
        self.games_won = 0
        
    def reset(self):
        """Reset the environment to initial state"""
        self.current_streak = 0
        self.last_3_results = [0, 0, 0]
        self.current_bet = 0
        self.balance = 100
        self.consecutive_odds = 0
        self.total_games = 0
        self.games_won = 0
        return self._get_observation()
    
    def _get_observation(self):
        """Convert current state to observation"""
        return np.array([
            self.current_streak,
            *self.last_3_results,
            self.current_bet,
            self.balance
        ], dtype=np.float32)
    
    def _toss_coins(self):
        """Simulate tossing two coins"""
        coin1 = random.randint(0, 1)  # 0: heads, 1: tails
        coin2 = random.randint(0, 1)
        
        if coin1 == coin2:
            self.consecutive_odds = 0
            return 0 if coin1 == 0 else 1  # 0: heads, 1: tails
        else:
            self.consecutive_odds += 1
            return 2  # odds
    
    def step(self, action):
        """
        Execute one time step within the environment
        
        Args:
            action: 0 for no bet, 1 for heads, 2 for tails
            
        Returns:
            observation: Current state
            reward: Reward for the action
            done: Whether episode is done
            info: Additional information
        """
        # Place bet (action)
        bet_amount = 10  # Fixed bet amount for simplicity
        
        if action == 0:  # No bet
            self.current_bet = 0
            reward = 0
            result = None
        else:
            self.current_bet = bet_amount
            # Toss coins
            result = self._toss_coins()
            
            # Update last 3 results
            self.last_3_results = self.last_3_results[1:] + [result]
            
            # Calculate reward
            reward = 0
            if result != 2:  # Not odds
                self.total_games += 1
                if (action == 1 and result == 0) or (action == 2 and result == 1):
                    reward = bet_amount  # Win
                    self.balance += bet_amount
                    self.current_streak += 1
                    self.games_won += 1
                else:
                    reward = -bet_amount  # Loss
                    self.balance -= bet_amount
                    self.current_streak = 0
        
        # Check if episode should end
        done = False
        if self.balance <= 0:
            done = True
            reward = -100  # Penalty for going broke
        elif self.consecutive_odds >= 3:
            done = True
            reward = -50  # Penalty for three consecutive odds
        
        info = {
            'balance': self.balance,
            'result': result,
            'consecutive_odds': self.consecutive_odds,
            'win_rate': self.games_won / max(1, self.total_games)
        }
        
        return self._get_observation(), reward, done, info
    
    def render(self, mode='human'):
        """Render the current state"""
        if mode == 'human':
            print(f"Balance: ${self.balance}")
            print(f"Current Streak: {self.current_streak}")
            print(f"Last 3 Results: {self.last_3_results}")
            print(f"Consecutive Odds: {self.consecutive_odds}")
            print(f"Win Rate: {self.games_won/max(1, self.total_games):.2%}")
            print("-" * 30) 