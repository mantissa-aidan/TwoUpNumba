# Two Up Environment Improvements

## 0. Personality-Based Agent Framework & Reporting (Recent Major Update)
- [x] Restructure project for personality-specific agent training, evaluation, and reporting.
  - [x] Models, TensorBoard logs, and reports saved to `personality_agents/<type>/`.
  - [x] Core training script (`train_personality_agent.py`) updated for personality focus.
  - [x] Ensured CPU-only operation for current training and evaluation tasks.
- [x] Refactor evaluation and comparison:
    - [x] `rule_based_strategies.py` now self-contained for benchmarking rule-based agents, outputs JSON report to `personality_agents/<type>/reports/rule_based_strategies_summary_<type>_<timestamp>.json`.
    - [x] `compare_personalities.py` refocused to compare PPO agent vs. Random and No-Bet baselines, outputs JSON report and plot to `personality_agents/<type>/reports/ppo_vs_baselines_<type>_<timestamp>.json/png`.
    - [x] **New script**: `generate_master_report.py` created to load JSONs from the above two scripts, merge summary data, and output a combined JSON (`master_summary_report_<type>_<timestamp>.json`) and plot (`master_summary_plot_<type>_<timestamp>.png`) to `personality_agents/<type>/reports/`.
- [ ] **Next Steps - Execution & Analysis:**
    - [ ] Train PPO agents for all defined personalities: 'standard', 'thrill_seeker', 'loss_averse', 'play_for_time' using `train_personality_agent.py`.
    - [ ] Run `rule_based_strategies.py` for each personality to generate their benchmarks.
    - [ ] Run `compare_personalities.py` for each trained PPO agent against baselines.
    - [ ] Run `generate_master_report.py` for each personality to get the final combined reports and plots.
    - [ ] Analyze the results from the master reports.
- [ ] **[IMPORTANT]** Resolve bet amount specification for rule-based strategies in `rule_based_strategies.py` (see `TODO` in file) to ensure they can dictate exact bet stakes if needed, potentially by modifying `TwoUpEnvNumba.step()`.

## 1. Migrate to Gymnasium ✅
- [x] Update imports from `gym` to `gymnasium`
- [x] Update environment class to inherit from `gymnasium.Env`
- [x] Update action and observation spaces to use Gymnasium's space classes
- [x] Update metadata and render modes to match Gymnasium standards
- [x] Add proper type hints and docstrings following Gymnasium conventions

## 2. GPU Support ✅
- [x] Add PyTorch as a dependency
- [x] Update requirements.txt with GPU-compatible versions
- [x] Modify PPO implementation to use GPU when available (Note: Current project phase focuses on CPU-only for personality agents, but underlying capability exists).
- [x] Add device detection and automatic GPU usage (currently overridden to CPU in scripts).
- [x] Add memory management for GPU operations

## 3. Training Visualization
- [x] Add TensorBoard integration (logs now personality-specific: `personality_agents/<type>/tensorboard_logs/`)
- [ ] Create custom logging for:
  - [ ] Training rewards (beyond TensorBoard defaults)
  - [ ] Win rates
  - [ ] Balance progression
  - [ ] Action distribution
- [ ] Add real-time plotting of training metrics
- [ ] Create episode replay visualization
- [x] Add policy visualization (action probabilities)

## 4. Simulation Visualization
- [ ] Create interactive web-based visualization using Streamlit
- [ ] Add features to:
  - [ ] Show real-time coin tosses
  - [ ] Display balance changes
  - [ ] Show betting decisions
  - [ ] Visualize win/loss streaks
- [ ] Add comparison charts for different strategies
- [ ] Create animated transitions between states

## 5. Code Structure Improvements
- [ ] Create proper package structure
- [ ] Add configuration management
- [ ] Implement proper logging system
- [ ] Add unit tests
- [ ] Add integration tests
- [ ] Create documentation

## 6. Performance Optimizations
- [x] Implement vectorized environments
- [x] Integrate Numba for environment core logic
- [ ] Add parallel simulation support
- [ ] Optimize data structures
- [ ] Add caching for frequently used calculations
- [ ] Profile and optimize bottlenecks

## 7. New Features
- [ ] Add variable bet sizes
- [ ] Implement different betting strategies
- [ ] Add multiplayer support
- [ ] Create tournament mode
- [ ] Add historical data analysis

## 8. Documentation
- [ ] Create API documentation
- [ ] Add usage examples
- [ ] Create installation guide
- [ ] Add contribution guidelines
- [ ] Create user manual

## 9. Deployment
- [ ] Create Docker container
- [ ] Set up CI/CD pipeline
- [ ] Add version management
- [ ] Create release process
- [ ] Add automated testing

## 10. Monitoring and Analysis
- [x] Add performance metrics (evaluation scripts now output detailed JSON reports to `personality_agents/<type>/reports/`)
- [x] Create analysis tools (`compare_personalities.py` for PPO vs baselines, `rule_based_strategies.py` for rule benchmarks, `generate_master_report.py` for combined overview)
- [ ] Add debugging tools
- [ ] Implement error tracking
- [ ] Add system monitoring

## 11. RL Agent Strategy Analysis
- [x] Train RL agent using PPO on Numba-optimized environment (Framework enhanced for multiple personalities).
- [x] Evaluate trained RL agent over a large number of episodes (Supported by `compare_personalities.py`).
- [x] Document key performance metrics (avg. final balance, win rate, avg. rounds) (Done via JSON reports from evaluation scripts).
- [ ] Analyze betting patterns and decision-making of the RL agent (Qualitative analysis pending, based on master reports and potentially episode replays).
- [x] Compare RL agent's strategy and performance against scripted strategies (Handled by `generate_master_report.py`).

## Priority Order
1. ~~Migrate to Gymnasium (Critical)~~ ✅
2. ~~Add GPU Support (High)~~ (✅ underlying, but current phase is CPU)
3. **Personality-Based Agent Framework - Execution & Bet Resolution (NEW - CRITICAL)**
   - [ ] Train all personality agents.
   - [ ] Run all evaluation and reporting scripts for each personality.
   - [ ] Resolve rule-based strategy bet amount issue in `rule_based_strategies.py` / `TwoUpEnvNumba.py`.
   - [ ] Analyze comprehensive results from master reports.
4. Training Visualization (High - for deeper insights beyond Tensorboard basics)
5. Code Structure Improvements (Medium)
6. Simulation Visualization (Medium)
7. Performance Optimizations (Medium)
8. Documentation (Medium)
9. New Features (Low)
10. Deployment (Low)
11. Monitoring and Analysis (Low)
12. RL Agent Strategy Analysis (Low)

## Notes
- Each task should be implemented in a separate branch
- All changes should be properly tested
- Documentation should be updated as features are added
- Performance metrics should be collected before and after changes
- Regular backups should be maintained 