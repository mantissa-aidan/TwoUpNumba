# PowerShell script to run the full pipeline for all personalities

[CmdletBinding()]
Param(
    [Switch]$ForceRetrain
)

# Define the personalities
$personalities = @("standard", "thrill_seeker", "loss_averse", "play_for_time")

# Define the number of episodes for evaluation scripts (applies to rule-based and PPO comparisons)
$numEpisodes = 1000

Write-Host "Starting full processing pipeline..."
if ($ForceRetrain) {
    Write-Host "ForceRetrain flag is set. All PPO models will be retrained even if they exist."
}
Write-Host "Ensure Python environment is active and scripts are in the current path or specify full paths."
Write-Host "=================================================================="

# Step 0: Benchmark Rule-Based Strategies (Universal - Run Once)
# This step is always run as it's relatively quick and its output might be updated.
Write-Host ""
Write-Host "------------------------------------------------------------------"
Write-Host "[Step 0 - Universal] Benchmarking rule-based strategies (Episodes: $numEpisodes)..."
Write-Host "------------------------------------------------------------------"
python rule_based_strategies.py --episodes $numEpisodes # Personality argument no longer needed
if ($LASTEXITCODE -ne 0) {
    Write-Error "Error during universal rule-based strategy benchmarking. Halting script."
    exit $LASTEXITCODE
}
Write-Host "[Step 0 - Universal] Finished benchmarking rule-based strategies."
Write-Host "Results saved to reports/rule_based_strategies_summary_universal_<timestamp>.json"
Write-Host "=================================================================="


foreach ($personality in $personalities) {
    Write-Host ""
    Write-Host "------------------------------------------------------------------"
    Write-Host "Processing PPO agent for personality: $personality"
    Write-Host "------------------------------------------------------------------"

    # Define model path
    $modelFile = "personality_agents/$personality/models/ppo_two_up_$personality.zip"

    # Step 1: Train PPO Agent for current personality (Conditional)
    if ($ForceRetrain -Or !(Test-Path $modelFile)) {
        Write-Host "[Step 1/3 for $personality] Training PPO agent..."
        # Ensure model directory exists before training
        $modelDir = "personality_agents/$personality/models"
        if (!(Test-Path $modelDir)) {
            New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
        }
        python train_personality_agent.py --personality $personality
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Error during training for $personality. Halting script."
            exit $LASTEXITCODE
        }
        Write-Host "[Step 1/3 for $personality] Finished training PPO agent."
    } else {
        Write-Host "[Step 1/3 for $personality] Skipping training. Model file already exists at '$modelFile'. Use -ForceRetrain to retrain."
    }
    Write-Host ""

    # Step 2: Evaluate PPO vs. Baselines for current personality
    # This step should run if the model exists (either newly trained or pre-existing)
    if (Test-Path $modelFile) {
        Write-Host "[Step 2/3 for $personality] Evaluating PPO vs. baselines (Episodes: $numEpisodes)..."
        python compare_personalities.py --personality $personality --episodes $numEpisodes
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Error during PPO vs. baselines evaluation for $personality. Halting script."
            exit $LASTEXITCODE
        }
        Write-Host "[Step 2/3 for $personality] Finished evaluating PPO vs. baselines."
    } else {
        Write-Warning "[Step 2/3 for $personality] Skipping PPO evaluation. Model file '$modelFile' not found and was not trained in this run."
    }
    Write-Host ""

    # Step 3: Generate Master Report for current personality (compares PPO to universal rule-based)
    # This step should also ideally only run if PPO data is available to compare.
    $ppoReportFilePattern = "personality_agents/$personality/reports/ppo_vs_baselines_*.json"
    if (Test-Path $modelFile -And (Get-ChildItem $ppoReportFilePattern -ErrorAction SilentlyContinue)) { # Check if model exists and PPO report was likely generated
        Write-Host "[Step 3/3 for $personality] Generating master report..."
        python generate_master_report.py --personality $personality
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Error during master report generation for $personality. Halting script."
            exit $LASTEXITCODE
        }
        Write-Host "[Step 3/3 for $personality] Finished generating master report."
    } else {
        Write-Warning "[Step 3/3 for $personality] Skipping master report generation. PPO model or its evaluation report not found for '$personality'."
    }
    Write-Host "------------------------------------------------------------------"
    Write-Host "Completed PPO agent processing for personality: $personality"
    Write-Host "=================================================================="
}

# Final Step: Run Break-Even Analysis (if reports exist)
Write-Host ""
Write-Host "------------------------------------------------------------------"
Write-Host "[Final Step] Running Break-Even Chance Analysis..."
Write-Host "------------------------------------------------------------------"
python analyze_breakeven_chances.py --plot
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Warning during break-even chance analysis (script might have handled missing files gracefully)."
    # Not exiting with error as analyze_breakeven_chances.py is designed to handle missing files.
}
Write-Host "[Final Step] Finished Break-Even Chance Analysis."
Write-Host "Plot saved to reports/analysis_results/breakeven_chance_comparison.png (if data was available)"
Write-Host "=================================================================="

Write-Host ""
Write-Host "All PPO personalities processed and compared against universal rule-based benchmarks."
Write-Host "Pipeline finished." 