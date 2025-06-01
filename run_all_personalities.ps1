# PowerShell script to run the full pipeline for all personalities

param ( 
    [Switch]$ForceRetrain 
) 

# Define global parameters for the runs
$InitialBalance = 100.0
$BaseBetUnit = 10.0      # This is the base_bet_unit for the env, used in betting calculations
$MaxEpisodeSteps = 200
$TotalTimestepsPPO = 100000 # Timesteps for PPO training
$NumEpisodesEval = 1000    # Episodes for evaluations (rule-based, PPO vs baselines)

$Personalities = @("standard", "thrill_seeker", "loss_averse", "play_for_time")

Write-Host "Starting the Two-Up RL Agent Pipeline..."
Write-Host "Global Run Parameters:
  Initial Balance: $InitialBalance
  Base Bet Unit: $BaseBetUnit
  Max Episode Steps: $MaxEpisodeSteps
  PPO Training Timesteps: $TotalTimestepsPPO
  Evaluation Episodes: $NumEpisodesEval"
if ($ForceRetrain) {
    Write-Host "ForceRetrain flag is SET. PPO models will be retrained even if they exist."
}
Write-Host "=================================================================="

# --- 1. Run Universal Rule-Based Strategies (Once) ---
Write-Host "`n--- [STEP 1/3] Running Universal Rule-Based Strategies ---"
python rule_based_strategies.py --num-episodes $NumEpisodesEval --initial-balance $InitialBalance --base-bet $BaseBetUnit --max-episode-steps $MaxEpisodeSteps
if ($LASTEXITCODE -ne 0) {
    Write-Error "Rule-based strategies script failed. Halting."
    exit 1
}
Write-Host "Universal Rule-Based Strategies evaluation complete."

# --- 2. Process Each PPO Personality ---
Write-Host "`n--- [STEP 2/3] Processing PPO Agent Personalities ---"
foreach ($personality in $Personalities) {
    Write-Host "`n  --- Processing Personality: $personality ---"
    $ErrorActionPreference = "Stop" # Stop script if a command fails within this personality block

    $modelFile = "personality_agents/$personality/models/ppo_two_up_$personality.zip"
    $ppoReportFileDir = "personality_agents/$personality/reports"
    
    # Check for existing PPO vs Baselines report *before* training/comparison attempts
    $existingPpoVsBaselinesReportPath = Get-ChildItem -Path $ppoReportFileDir -Filter "ppo_vs_baselines_${personality}_*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { $_.FullName }

    # a. Train PPO Agent
    if ($ForceRetrain -or !(Test-Path $modelFile)) {
        Write-Host "    [2a] Training PPO agent for $personality..."
        python train_personality_agent.py --personality $personality --timesteps $TotalTimestepsPPO --initial-balance $InitialBalance --base-bet $BaseBetUnit --max-episode-steps $MaxEpisodeSteps
        if ($LASTEXITCODE -ne 0) {
            Write-Error "    PPO training for $personality failed. Skipping subsequent steps for this personality."
            continue # Skip to next personality
        }
        Write-Host "    PPO training for $personality completed."
    } else {
        Write-Host "    [2a] Skipping PPO training for $personality (model exists: $modelFile and -ForceRetrain not used)."
    }

    # b. Compare PPO Agent vs. Baselines (only if model exists)
    if (Test-Path $modelFile) {
        Write-Host "    [2b] Comparing PPO agent vs. baselines for $personality..."
        python compare_personalities.py --personality $personality --num-episodes $NumEpisodesEval --initial-balance $InitialBalance --base-bet $BaseBetUnit --max-episode-steps $MaxEpisodeSteps
        if ($LASTEXITCODE -ne 0) {
            Write-Error "    PPO comparison for $personality failed. Skipping master report for this personality."
            # Update path to nullify, so master report doesn't try to use potentially partial/old data
            $existingPpoVsBaselinesReportPath = $null 
            continue # Skip to next personality if comparison fails critically
        }
        Write-Host "    PPO comparison for $personality completed."
        # Refresh the path to ensure we get the newly created one
        $existingPpoVsBaselinesReportPath = Get-ChildItem -Path $ppoReportFileDir -Filter "ppo_vs_baselines_${personality}_*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { $_.FullName }
    } else {
        Write-Host "    [2b] Skipping PPO comparison for $personality (model file not found: $modelFile)."
        $existingPpoVsBaselinesReportPath = $null # Ensure no master report if PPO model is missing
    }

    # c. Generate Master Report (only if PPO vs Baselines report exists for this personality)
    if ($existingPpoVsBaselinesReportPath -and (Test-Path $existingPpoVsBaselinesReportPath)) {
        Write-Host "    [2c] Generating master summary report for $personality..."
        python generate_master_report.py --personality $personality --initial-balance $InitialBalance --base-bet $BaseBetUnit --max-episode-steps $MaxEpisodeSteps --num-episodes-context $NumEpisodesEval
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "    Master report generation for $personality encountered an issue (non-critical)."
            # Continue, as this is a reporting step
        } else {
            Write-Host "    Master report for $personality generated."
        }
    } else {
        Write-Host "    [2c] Skipping master report generation for $personality (PPO vs Baselines report not found or comparison failed: $existingPpoVsBaselinesReportPath)."
    }
    $ErrorActionPreference = "Continue" # Reset to default error action preference
}
Write-Host "`nAll PPO agent personalities processed."

# --- 3. Final Analysis (Break-Even Chances) ---
Write-Host "`n--- [STEP 3/3] Analyzing Break-Even Chances Across All Strategies ---"
python analyze_breakeven_chances.py --plot --initial-balance $InitialBalance
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Break-even analysis script encountered an issue."
}
Write-Host "Break-even analysis complete."

Write-Host "`n=================================================================="
Write-Host "Two-Up RL Agent Variable Bet Size Pipeline finished."
Write-Host "Check the 'personality_agents/<personality>/reports' and 'reports/analysis_results' directories for outputs." 