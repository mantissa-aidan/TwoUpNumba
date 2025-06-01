import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numba import jit, int32, float32, boolean
from typing import Tuple, Dict, Any, Optional

@jit(nopython=True)
def toss_coins_numba(rng_state):
    """Numba-optimized coin toss function"""
    # Corrected random integer generation for Numba compatibility if needed,
    # assuming np.random.RandomState object is passed if seeding is critical.
    # For simplicity here, using the global np.random which might not be ideal for strict reproducibility in Numba.
    # A better approach would be to pass a Numba-compatible random generator state.
    coin1 = np.random.randint(0, 2) 
    coin2 = np.random.randint(0, 2)
    
    if coin1 == coin2:
        return 0 if coin1 == 0 else 1, 0  # 0: heads, 1: tails, 0: no consecutive odds
    else:
        return 2, 1  # 2: odds, 1: consecutive odds

@jit(nopython=True)
def get_observation_numba(current_streak, last_3_results_flat_0, last_3_results_flat_1, last_3_results_flat_2, current_bet, balance): # Modified for individual elements
    return np.array([
        current_streak,
        last_3_results_flat_0, # Pass individual elements
        last_3_results_flat_1,
        last_3_results_flat_2,
        current_bet,
        balance
    ], dtype=np.float32)

class TwoUpEnvNumba(gym.Env):
    """Numba-optimized Two Up Environment"""
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, render_mode: Optional[str] = None, gambler_personality: str = 'standard', initial_balance: float = 100.0, base_bet_amount: float = 5.0, max_episode_steps: int = 200):
        super().__init__()
        
        self.fixed_bet_amount = np.float32(base_bet_amount) # Use passed base_bet_amount
        self.initial_balance = np.float32(initial_balance) # Use passed initial_balance
        self.gambler_personality = gambler_personality
        self.max_steps = max_episode_steps # Store max_episode_steps
        self.current_step_in_episode = 0 # Initialize step counter for truncation
        
        # Action: 0: No Bet, 1: Bet Heads, 2: Bet Tails
        self.action_space = spaces.Discrete(3) 
        
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32), # Added dtype
            high=np.array([10, 2, 2, 2, self.fixed_bet_amount, 1000], dtype=np.float32), # Corrected last_3_results and current_bet high
            dtype=np.float32
        )
        
        # Game state
        self.current_streak = np.int32(0)
        self.last_3_results = np.zeros(3, dtype=np.int32) # Ensure it's int32 for Numba
        self.current_bet = np.float32(0.0)
        self.balance = np.float32(100.0)
        self.consecutive_odds = np.int32(0) # Still tracked for info, not agent observation
        self.total_games = np.int32(0)
        self.games_won = np.int32(0)
        self.render_mode = render_mode
        # Initialize the random number generator state for Numba
        # This is a simplified way; for true seeded Numba randomness, 
        # you'd manage and pass the np.random.RandomState._bitgen explicitly if possible or use a Numba-specific PRNG.
        # However, np.random.seed() in the main script often suffices for top-level seeding.
    
    def _get_observation(self):
        """Get current observation using Numba-optimized function"""
        return get_observation_numba(
            np.float32(self.current_streak), # Ensure types match Numba function signature
            np.int32(self.last_3_results[0]), # Pass individual elements and ensure type
            np.int32(self.last_3_results[1]),
            np.int32(self.last_3_results[2]),
            np.float32(self.current_bet),
            np.float32(self.balance)
        )
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        if seed is not None:
            # Numba uses its own PRNG state. Seeding numpy globally might not directly control
            # the randomness inside JITted functions perfectly across all scenarios.
            # For reproducible Numba randomness, more care is needed, possibly by passing
            # a seeded generator or using a Numba-specific way to seed.
            np.random.seed(seed)

        self.current_streak = np.int32(0)
        self.last_3_results = np.zeros(3, dtype=np.int32)
        self.current_bet = np.float32(0.0)
        self.balance = self.initial_balance # Reset to initial_balance
        self.consecutive_odds = np.int32(0)
        self.total_games = np.int32(0)
        self.games_won = np.int32(0)
        self.current_step_in_episode = 0 # Reset step counter
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment"""
        terminated = False # Initialize terminated
        truncated = False # Initialize truncated
        self.current_step_in_episode += 1 # Increment step counter
        
        # Toss coins first, as it happens regardless of bet (to update history)
        # Pass the environment's random number generator state to the Numba function
        # For Numba to use the gym's seeded RNG, direct passing of self.np_random might be complex.
        # toss_coins_numba is currently using global np.random.
        # If strict seeding for toss_coins_numba is required from gym's seed, 
        # this part needs Numba-compatible RNG state management.
        result, odds_increment = toss_coins_numba(self.np_random) # self.np_random is gym's seeded generator
        
        # Update history based on the toss
        self.last_3_results = np.roll(self.last_3_results, -1)
        self.last_3_results[-1] = result
        if result == 2: # odds
             self.consecutive_odds += 1
        else:
             self.consecutive_odds = 0 # reset if not odds

        reward = np.float32(0.0) # Initialize reward as float
        
        # Personality-based initial reward adjustments
        if self.gambler_personality == 'play_for_time':
            reward += np.float32(0.05) # Survival bonus per step

        if action == 0:  # No bet
            self.current_bet = np.float32(0.0)
            if self.gambler_personality == 'thrill_seeker':
                reward -= np.float32(0.1) # Small penalty for inaction
        else:  # Bet placed (action 1 for Heads, action 2 for Tails)
            if self.gambler_personality == 'thrill_seeker':
                reward += np.float32(0.2) # Small bonus for placing a bet

            if self.fixed_bet_amount > self.balance:
                reward += np.float32(-10.0) if self.gambler_personality != 'thrill_seeker' else np.float32(-5.0)
                self.current_bet = np.float32(0.0)
            else:
                self.current_bet = self.fixed_bet_amount
                if result != 2:  # Bet resolves only if not odds
                    self.total_games += 1
                    # Action 1 is Bet Heads (result 0 is Heads)
                    # Action 2 is Bet Tails (result 1 is Tails)
                    is_win = (action == 1 and result == 0) or \
                             (action == 2 and result == 1)
                    
                    if is_win:
                        win_amount = self.fixed_bet_amount
                        reward += win_amount
                        self.balance += win_amount
                        self.current_streak += 1
                        self.games_won += 1
                    else:
                        loss_amount = self.fixed_bet_amount
                        if self.gambler_personality == 'loss_averse':
                            loss_amount *= np.float32(1.5)
                        reward -= loss_amount
                        self.balance -= self.fixed_bet_amount
                        self.current_streak = 0
                # If it's odds (result == 2), the bet is a "push" or no action, reward is 0
                # unless a specific house rule for odds bets is implemented. 
                # Current logic: no win/loss on odds if a bet was placed.
        
        terminated = self.balance <= 0
        if terminated:
            if self.gambler_personality == 'standard': reward = np.float32(-100.0)
            elif self.gambler_personality == 'thrill_seeker': reward = np.float32(-50.0)
            elif self.gambler_personality == 'loss_averse': reward = np.float32(-150.0)
            elif self.gambler_personality == 'play_for_time': reward = np.float32(-100.0) # Still penalize ruin
            else: reward = np.float32(-100.0) # Default for unknown personality

        # Check for truncation due to max_episode_steps
        if not terminated and self.current_step_in_episode >= self.max_steps:
            truncated = True
            # Optional: Add a specific reward adjustment for truncation if desired
            # reward += np.float32(0.0) # Or some other value

        info = {
            'balance': np.float32(self.balance), # Ensure float
            'result': np.int32(result) if result is not None else -1, # Ensure int, handle None
            'consecutive_odds': np.int32(self.consecutive_odds), # Ensure int
            'win_rate': np.float32(self.games_won / max(1, self.total_games)) if self.total_games > 0 else np.float32(0.0), # Ensure float
            'reason': 'broke' if terminated else ('max_steps' if truncated else None),
            'total_games': np.int32(self.total_games) # Added for consistency
        }
        
        return self._get_observation(), reward, terminated, truncated, info # Return truncated
    
    def render(self) -> Optional[np.ndarray]:
        """Render the current state"""
        if self.render_mode == 'human':
            print(f"Balance: ${self.balance:.2f}, Streak: {self.current_streak}, Last 3: {self.last_3_results}, Cons. Odds: {self.consecutive_odds}, WG: {self.games_won}/{self.total_games}")
        elif self.render_mode == 'rgb_array':
            # TODO: Implement RGB array rendering
            return None 