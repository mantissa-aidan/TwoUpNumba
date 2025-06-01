import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
from typing import Tuple, Dict, Any, Optional

class TwoUpEnv(gym.Env):
    """
    Two Up Environment for Reinforcement Learning.
    
    This environment simulates the Two Up coin game where two coins are tossed
    and players can bet on the outcome (heads, tails, or no bet).
    
    Attributes:
        action_space: Discrete space with 3 actions (0: no bet, 1: bet on heads, 2: bet on tails)
        observation_space: Box space containing [current_streak, last_3_results, current_bet_amount, balance]
        metadata: Environment metadata including supported render modes
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, render_mode: Optional[str] = None):
        """
        Initialize the Two Up environment.
        
        Args:
            render_mode: The render mode to use ('human' or 'rgb_array')
        """
        super().__init__()
        
        # Define action space (0: no bet, 1: bet $5, 2: bet $10, 3: bet $20, 4: bet $50, 5: bet $100, 6: quit)
        self.action_space = spaces.Discrete(7)
        
        # Define observation space
        # [current_streak, last_3_results, current_bet_amount, balance]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([10, 1, 1, 1, 100, 1000]),
            dtype=np.float32
        )
        
        # Game state
        self.current_streak = 0
        self.last_3_results = [0, 0, 0]  # 0: heads, 1: tails, 2: odds
        self.current_bet = 0
        self.balance = 100  # Starting balance
        self.consecutive_odds = 0  # Keep track for observation, but don't use for termination
        self.total_games = 0
        self.games_won = 0
        self.render_mode = render_mode
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: The seed for the random number generator
            options: Additional options for reset
            
        Returns:
            observation: The initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        self.current_streak = 0
        self.last_3_results = [0, 0, 0]
        self.current_bet = 0
        self.balance = 100
        self.consecutive_odds = 0
        self.total_games = 0
        self.games_won = 0
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """
        Convert current state to observation.
        
        Returns:
            np.ndarray: The current observation
        """
        return np.array([
            self.current_streak,
            *self.last_3_results,
            self.current_bet,
            self.balance
        ], dtype=np.float32)
    
    def _toss_coins(self) -> int:
        """
        Simulate tossing two coins.
        
        Returns:
            int: 0 for heads, 1 for tails, 2 for odds
        """
        coin1 = self.np_random.integers(0, 2)  # 0: heads, 1: tails
        coin2 = self.np_random.integers(0, 2)
        
        if coin1 == coin2:
            self.consecutive_odds = 0
            return 0 if coin1 == 0 else 1  # 0: heads, 1: tails
        else:
            self.consecutive_odds += 1
            return 2  # odds
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.
        
        Args:
            action: 0 for no bet, 1-5 for different bet amounts, 6 for quit
            
        Returns:
            observation: Current state
            reward: Reward for the action
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Check for quit action
        if action == 6:
            return self._get_observation(), 0, True, False, {
                'balance': self.balance,
                'result': None,
                'consecutive_odds': self.consecutive_odds,
                'win_rate': self.games_won / max(1, self.total_games),
                'reason': 'voluntary_quit'
            }

        # Place bet (action)
        bet_amounts = [0, 5, 10, 20, 50, 100]
        bet_amount = bet_amounts[action]

        if bet_amount > self.balance:
            reward = -10  # Penalty for trying to bet more than balance
            result = None
        else:
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
        terminated = False
        truncated = False
        end_reason = None
        if self.balance <= 0:
            terminated = True
            reward = -100  # Penalty for going broke
            end_reason = 'broke'
        
        info = {
            'balance': self.balance,
            'result': result,
            'consecutive_odds': self.consecutive_odds,
            'win_rate': self.games_won / max(1, self.total_games),
            'reason': end_reason
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the current state.
        
        Returns:
            Optional[np.ndarray]: RGB array if render_mode is 'rgb_array', None otherwise
        """
        if self.render_mode == 'human':
            print(f"Balance: ${self.balance}")
            print(f"Current Streak: {self.current_streak}")
            print(f"Last 3 Results: {self.last_3_results}")
            print(f"Consecutive Odds: {self.consecutive_odds}")
            print(f"Win Rate: {self.games_won/max(1, self.total_games):.2%}")
            print("-" * 30)
        elif self.render_mode == 'rgb_array':
            # TODO: Implement RGB array rendering
            return None 