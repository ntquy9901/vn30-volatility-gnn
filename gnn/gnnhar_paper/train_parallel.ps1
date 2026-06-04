# Train GNN-HAR for all horizons in parallel
# Usage: .\train_parallel.ps1

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
cd $scriptDir

Write-Host "Starting parallel training for all horizons..." -ForegroundColor Green

# Start background jobs for each horizon
Start-Job -ScriptBlock {
    cd $args[0]
    python train_gnnhar_paper.py --horizon 1
} -ArgumentList $scriptDir -Name "h1"

Start-Job -ScriptBlock {
    cd $args[0]
    python train_gnnhar_paper.py --horizon 5
} -ArgumentList $scriptDir -Name "h5"

Start-Job -ScriptBlock {
    cd $args[0]
    python train_gnnhar_paper.py --horizon 10
} -ArgumentList $scriptDir -Name "h10"

Start-Job -ScriptBlock {
    cd $args[0]
    python train_gnnhar_paper.py --horizon 20
} -ArgumentList $scriptDir -Name "h20"

Write-Host "All jobs started. Check status with: Get-Job" -ForegroundColor Cyan
Write-Host "View output: Receive-Job -Name h1 | Select -Expand Output" -ForegroundColor Cyan

# Optional: Wait for all jobs (commented out - let them run in background)
# Wait-Job -Name "h1","h5","h10","h20"
# Write-Host "All training complete!" -ForegroundColor Green
