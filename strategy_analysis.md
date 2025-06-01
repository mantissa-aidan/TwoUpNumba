# Two-Up Game Strategy Analysis

## Summary Analysis: Scripted Two-Up Betting Strategies

**Objective:** To identify effective betting strategies in the Two-Up game by simulating various pre-defined approaches over 100,000 episodes each. Each episode had a maximum of 200 rounds, and players started with $100.

**Key Findings:**

1.  **No Guaranteed Profit Strategy:**
    *   None of the tested strategies demonstrated a reliable way to generate significant profit.
    *   The **"Always Heads ($20 base)"** strategy was the only one to finish with an average final balance slightly above the starting $100 (ending with $100.63 on average). However, this very marginal gain came with considerable risk, as indicated by a very negative "Avg Total Reward" ($-48.63), suggesting many instances of going broke.
    *   Most strategies resulted in an average final balance at or slightly below the starting $100.

2.  **Risk vs. Reward Trade-off:**
    *   **Highest Potential Reward (Extremely High Risk):** The **"Legend (All In Heads)"** strategy, where players bet their entire balance on Heads each round, showed the highest *potential* for large gains (Avg Max Balance: $1286.32, Avg Total Reward: $1086.32). However, it was by far the riskiest, with an average final balance of $0.00 and the shortest average game duration (4 rounds). This strategy is akin to a lottery ticket – a tiny chance of a huge win, but an overwhelming likelihood of losing everything quickly.
    *   **Lowest Risk (No Reward):** The **"Dickhead Strategy (No Bet)"** was the safest, guaranteeing the preservation of the initial $100 by never betting. Consequently, it offered zero financial reward but allowed players to "play" for the maximum 200 rounds.

3.  **Performance of "Intelligent" Strategies:**
    *   **"Pattern Betting"** ($10 base showed slightly better results than $20 base): This strategy, which tried to identify and bet on trends (sequences of heads/tails or alternations), achieved the highest average win rate on individual bets (49.3% for $10 base). However, this did not translate into overall profitability, with an average final balance of $99.98. Its "Avg Total Reward" was very close to zero ($-0.03), suggesting it was relatively stable and good at avoiding ruin, partly due to its rule of quitting if no pattern emerged.
    *   **"Streak Betting"**: This strategy, which adjusted bet sizes based on winning or losing streaks, did not yield positive average returns. It led to higher average maximum balances than simple "Always Heads/Tails" (e.g., $245.20 for $20 base) but also resulted in more negative "Avg Total Reward," indicating volatility and frequent losses.

4.  **Simple vs. Complex:**
    *   The simple **"Always Heads ($20 base)"** outperformed more complex strategies like "Pattern Betting" and "Streak Betting" in terms of average final balance, albeit marginally.
    *   The completely random **"Vibes Based"** strategy performed similarly to always betting on heads or tails, ending with an average balance around $100 but with negative average total rewards.

5.  **Game Duration:**
    *   Strategies involving aggressive betting (e.g., "Legend," higher base bets in "Streak Betting" or "Always Heads/Tails") generally led to shorter game durations.
    *   Conservative or pattern-based strategies that included options to not bet or quit (like "Pattern Betting" or "Dickhead") allowed for longer average gameplay.

**Overall Conclusion for Scripted Strategies:**

The simulations suggest that in this Two-Up environment, much like real-world games of pure chance, it's exceptionally difficult to find a betting strategy that consistently yields profits. Strategies that aim for high rewards invariably come with high risks of ruin. More conservative or "intelligent" pattern-seeking strategies, while potentially offering better win rates on individual bets or longer playtimes, did not demonstrate an ability to overcome the inherent randomness and (likely) slight house edge of the game to produce a positive average return. The safest approach was not to bet at all, while the only strategy showing a (negligible) average profit was still quite risky.

---

## RL Agent Strategy Comparison

**(To be filled in after training and evaluating the RL agent)**

1.  **RL Agent Training Configuration:**
    *   Algorithm: PPO (`MlpPolicy`)
    *   Environment: `TwoUpEnvNumba`
    *   Total Timesteps: `100,000` (as per `train_agent.py`)
    *   Key Hyperparameters (from `stable-baselines3` defaults unless specified otherwise in `train_agent.py`):
        *   Learning Rate:
        *   Batch Size:
        *   Number of Epochs:
        *   Gamma (Discount Factor):
        *   GAE Lambda:
        *   Clipping Factor (PPO):
        *   Number of parallel environments: 4

2.  **RL Agent Performance Metrics (Based on `test_agent.py` execution over X episodes):**
    *   **Average Final Balance:** $______
    *   **Standard Deviation of Final Balance:** $______
    *   **Median Final Balance:** $______
    *   **Average Total Reward (from environment):** ______
    *   **Average Win Rate (on bets placed):** ______%
    *   **Average Rounds Played per Episode:** ______
    *   **Frequency of "Broke" State:** ______%
    *   **Frequency of "Voluntary Quit" (if action 6 is used):** ______%
    *   **Frequency of Reaching Max Rounds:** ______%
    *   **Average Max Balance Achieved During Episodes:** $______

3.  **Observed Betting Behavior & Learned Strategy:**
    *   *(Describe the typical actions taken by the agent. Does it prefer certain bets? Does it avoid betting frequently? Does its strategy appear to change based on its balance, streak, or last results? Is it aggressive, conservative, or does it exhibit a specific pattern?)*

4.  **Comparison with Scripted Strategies:**
    *   **Profitability:** How does its average final balance compare to "Always Heads ($20 base)" and other top scripted contenders?
    *   **Risk Profile:** Is it more or less risky than strategies like "Legend" or "Streak Betting" (consider volatility, frequency of ruin)?
    *   **Stability/Consistency:** Is its performance more consistent than volatile scripted strategies? How does its "Avg Total Reward" compare?
    *   **Win Rate:** How does its bet win rate compare?
    *   **Play Style:** Does it resemble any of the scripted strategies, or has it learned something unique?

5.  **Conclusion on RL Agent's Effectiveness:**
    *   *(Summarize whether the RL agent learned a significantly better, worse, or different strategy compared to the predefined ones. Is it a viable approach for playing this game?)*

--- 