import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numba import jit
from typing import Optional, Tuple, Dict, Any, List

@jit(nopython=True)
def toss_coins_numba() -> Tuple[int, int, int]:
    """Numba-optimized coin toss function.
    Returns: (result, coin1, coin2)
    result: 0 for Heads (HH), 1 for Tails (TT), 2 for Odds (HT/TH)
    coin1, coin2: 0 for H, 1 for T
    """
    coin1 = np.random.randint(0, 2)
    coin2 = np.random.randint(0, 2)
    if coin1 == 0 and coin2 == 0:
        return 0, 0, 0  # Heads, Heads
    elif coin1 == 1 and coin2 == 1:
        return 1, 1, 1  # Tails, Tails
    else:
        return 2, coin1, coin2  # Odds

def toss_coins_readable() -> Tuple[str, str, str]:
    """Readable version of toss_coins_numba."""
    numba_res, c1, c2 = toss_coins_numba()
    res_map = {0: "Heads", 1: "Tails", 2: "Odds"}
    coin_map = {0: "Heads", 1: "Tails"}
    return res_map[numba_res], coin_map[c1], coin_map[c2]

class TwoUpEnvNumba(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, initial_balance:float=100.0, base_bet_amount:float=10.0,
                 gambler_personality:str='standard', max_episode_steps:int=200,
                 render_mode: Optional[str]=None):
        super().__init__()
        
        self.base_bet_unit = float(base_bet_amount)
        self.bet_multipliers = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        self.num_bet_levels = len(self.bet_multipliers)

        self.initial_balance = np.float32(initial_balance)
        if self.initial_balance <= 0:
            raise ValueError("Initial balance must be positive.")
            
        self.gambler_personality = gambler_personality
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        
        self.action_space = spaces.Discrete(1 + (2 * self.num_bet_levels))
        
        max_possible_bet_value = self.base_bet_unit * np.max(self.bet_multipliers)
        
        obs_low = np.array([
            0.0,  # rounds_played_ratio
            -1.0, # toss result (None)
            -1.0, 
            -1.0, 
            0.0,  # last_bet_amount / initial_balance (no bet)
            0.0   # current_balance / initial_balance (bankrupt)
        ], dtype=np.float32)
        
        obs_high = np.array([
            1.0,  # rounds_played_ratio
            2.0,  # toss result (Odds)
            2.0,  
            2.0,  
            max_possible_bet_value / self.initial_balance if self.initial_balance > 0 else 1.0,
            10.0  # current_balance can be 10x initial_balance
        ], dtype=np.float32)

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        
        self.balance: float = 0.0
        self.current_streak: int = 0
        self.games_won: int = 0
        self.total_games: int = 0
        self.current_step_in_episode: int = 0
        self.last_3_results_hr: List[str] = []
        self.last_3_results_numba: np.ndarray = np.full(3, -1, dtype=np.int32)
        self.last_bet_amount: float = 0.0
        self.termination_reason: str = ""

    def _get_numeric_toss_result(self, readable_result: str) -> int:
        if readable_result == "Heads": return 0
        if readable_result == "Tails": return 1
        return 2 # Odds

    def _convert_hr_results_to_numba_array(self, hr_results: List[str]) -> np.ndarray:
        numba_arr = np.full(3, -1, dtype=np.int32)
        for i, hr_res in enumerate(reversed(hr_results[-3:])):
            if i < 3:
                numba_arr[2 - i] = self._get_numeric_toss_result(hr_res)
        return numba_arr

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(6, dtype=np.float32)
        obs[0] = np.float32(self.current_step_in_episode / self.max_episode_steps if self.max_episode_steps > 0 else 0)
        obs[1] = np.float32(self.last_3_results_numba[2])
        obs[2] = np.float32(self.last_3_results_numba[1])
        obs[3] = np.float32(self.last_3_results_numba[0])
        obs[4] = np.float32(self.last_bet_amount / self.initial_balance if self.initial_balance > 0 else 0)
        obs[5] = np.float32(self.balance / self.initial_balance if self.initial_balance > 0 else 0)
        return obs

    def _get_info(self) -> Dict[str, Any]:
        return {
            "balance": self.balance,
            "current_streak": self.current_streak,
            "games_won": self.games_won,
            "total_games": self.total_games,
            "current_step_in_episode": self.current_step_in_episode,
            "last_3_results_hr": list(self.last_3_results_hr),
            "last_3_results_numba": self.last_3_results_numba.tolist(),
            "last_bet_amount": self.last_bet_amount,
            "win_rate": self.games_won / max(1, self.total_games) if self.total_games > 0 else 0.0,
            "termination_reason": self.termination_reason
        }

    def _calculate_reward(self, bet_type: int, bet_amount_this_step: float, numeric_toss_result: int, won_bet: bool) -> float:
        base_reward = 0.0
        if bet_type == 0:
            if self.gambler_personality == 'play_for_time': base_reward = 0.1
        else:
            if numeric_toss_result == 2:
                if self.gambler_personality == 'loss_averse': base_reward = -0.05
                elif self.gambler_personality == 'play_for_time': base_reward = 0.05
            elif won_bet:
                if self.gambler_personality == 'thrill_seeker': base_reward = bet_amount_this_step * 2.0
                elif self.gambler_personality == 'loss_averse': base_reward = bet_amount_this_step * 0.5 
                else: base_reward = bet_amount_this_step 
            else:
                if self.gambler_personality == 'thrill_seeker': base_reward = -bet_amount_this_step * 1.0
                elif self.gambler_personality == 'loss_averse': base_reward = -bet_amount_this_step * 2.0
                else: base_reward = -bet_amount_this_step
        
        if self.balance <= 0 and bet_type > 0:
            if self.gambler_personality == 'loss_averse': base_reward -= 50.0
            elif self.gambler_personality == 'thrill_seeker': base_reward -= 5.0
            else: base_reward -= 10.0
        
        return float(base_reward)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.balance = float(self.initial_balance)
        self.current_streak = 0
        self.games_won = 0
        self.total_games = 0
        self.current_step_in_episode = 0
        self.last_3_results_hr = [] 
        self.last_3_results_numba = np.full(3, -1, dtype=np.int32) 
        self.last_bet_amount = 0.0
        self.termination_reason = ""
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step_in_episode += 1
        terminated = False
        truncated = False
        
        bet_type = 0
        chosen_multiplier = 0.0
        bet_amount_this_step = 0.0
        won_bet_flag = False

        if action == 0:
            bet_type = 0
        elif 1 <= action <= self.num_bet_levels:
            bet_type = 1
            multiplier_index = action - 1
            chosen_multiplier = self.bet_multipliers[multiplier_index]
        elif (self.num_bet_levels + 1) <= action <= (1 + 2 * self.num_bet_levels - 1):
            bet_type = 2
            multiplier_index = action - (self.num_bet_levels + 1)
            chosen_multiplier = self.bet_multipliers[multiplier_index]

        if bet_type > 0:
            calculated_bet = self.base_bet_unit * chosen_multiplier
            bet_amount_this_step = min(max(0.01, calculated_bet), self.balance)
            if calculated_bet <=0:
                 bet_amount_this_step = 0.0
            if bet_amount_this_step <= 0:
                bet_type = 0
                bet_amount_this_step = 0.0
        
        self.last_bet_amount = bet_amount_this_step

        readable_toss, _, _ = toss_coins_readable()
        self.last_3_results_hr.append(readable_toss)
        if len(self.last_3_results_hr) > 3: self.last_3_results_hr.pop(0)
        self.last_3_results_numba = self._convert_hr_results_to_numba_array(self.last_3_results_hr)
        numeric_toss_result = self._get_numeric_toss_result(readable_toss)

        if bet_type > 0:
            self.total_games += 1
            if numeric_toss_result != 2:
                is_win = (bet_type == 1 and numeric_toss_result == 0) or \
                         (bet_type == 2 and numeric_toss_result == 1)
                if is_win:
                    self.balance += bet_amount_this_step
                    self.games_won += 1
                    self.current_streak += 1
                    won_bet_flag = True
                else:
                    self.balance -= bet_amount_this_step
                    self.current_streak = 0
        
        reward = self._calculate_reward(bet_type, bet_amount_this_step, numeric_toss_result, won_bet_flag)

        if self.balance <= 0.001:
            self.balance = 0.0
            terminated = True
            if not self.termination_reason: self.termination_reason = "bankrupt"
        
        if self.current_step_in_episode >= self.max_episode_steps:
            truncated = True
            if not terminated: self.termination_reason = "max_steps"

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == 'human':
            print(f"Step: {self.current_step_in_episode}, Balance: {self.balance:.2f}, Last Bet: {self.last_bet_amount:.2f}, Streak: {self.current_streak}, Last 3: {self.last_3_results_hr}")
        elif self.render_mode == 'rgb_array':
            return np.zeros((3, 100, 100), dtype=np.uint8)
        return None

    def close(self):
        pass

if __name__ == '__main__':
    env = TwoUpEnvNumba(initial_balance=100, base_bet_amount=10, gambler_personality='standard', render_mode='human')
    obs, info = env.reset()
    print(f"Initial Obs: {obs}, Info: {info}")
    print(f"Action Space: {env.action_space}, Sample: {env.action_space.sample()}")

    for i in range(15):
        action = env.action_space.sample()
        print(f"\n--- Round {i+1} --- Action: {action} ---")
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"Obs: {obs}")
        print(f"Reward: {reward:.2f}")
        print(f"Info: {info}")
        if terminated or truncated:
            print(f"Episode finished after {i+1} steps. Reason: {info.get('termination_reason')}")
            obs, info = env.reset()
            print(f"Environment reset. New Obs: {obs}, Info: {info}") 