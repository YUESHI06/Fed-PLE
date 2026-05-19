param(
    [string]$Device = "cuda:0",
    [int]$Seed = 42,
    [string]$DataDir = "./data",
    [string]$SmartBugsRaw = "../SmartBugs-Wild",
    [string]$SmartBugsData = "./data_smartbugs"
)

$ErrorActionPreference = "Stop"
$vuls = @("reentrancy", "integer_overflow", "time_dependency", "dos_failed_call")
$methods = @("Fed_RESCUER", "Fed_FedAvg", "Fed_CL", "Fed_CLC", "Fed_FedCorr", "Fed_ARFL")

if (-not (Test-Path "$DataDir/reentrancy")) {
    Write-Host "=== Processing DAppSCAN dataset ==="
    python data_processing/dappscan_processor.py `
        --dappscan_dir ../DAppSCAN-main/DAppSCAN-source `
        --output_dir $DataDir `
        --vul all `
        --seed $Seed
}

if ((Test-Path $SmartBugsRaw) -and (-not (Test-Path "$SmartBugsData/reentrancy"))) {
    Write-Host "=== Processing SmartBugs-Wild dataset ==="
    python data_processing/smartbugs_processor.py `
        --smartbugs_dir $SmartBugsRaw `
        --output_dir $SmartBugsData `
        --vul all `
        --seed $Seed
}

Write-Host "=== Task 3: Motivation Experiments ==="
python non_Fed_Train.py `
    --data_dir $DataDir `
    --smartbugs_data_dir $SmartBugsData `
    --device $Device `
    --seed $Seed

Write-Host "=== Task 4: Baseline Comparison ==="
foreach ($vul in $vuls) {
    foreach ($method in $methods) {
        python "$method.py" --vul $vul --data_dir $DataDir --device $Device --seed $Seed --noise_type pure --noise_rate 0.0 --epoch 30
        foreach ($nr in @(0.1, 0.2, 0.3)) {
            python "$method.py" --vul $vul --data_dir $DataDir --device $Device --seed $Seed --noise_type fn_noise --noise_rate $nr --epoch 30
        }
    }

    foreach ($method in $methods) {
        python "$method.py" --vul $vul --data_dir $DataDir --device $Device --seed $Seed --noise_type diff_noise --noise_rates 0.1 0.2 --epoch 30
        python "$method.py" --vul $vul --data_dir $DataDir --device $Device --seed $Seed --noise_type diff_noise --noise_rates 0.1 0.3 --epoch 30
    }
}

Write-Host "=== Task 5: Ablation Experiments ==="
foreach ($nr in @(0.0, 0.1, 0.2, 0.3)) {
    $nt = "fn_noise"
    if ($nr -eq 0.0) {
        $nt = "pure"
    }
    python Ablation.py --data_dir $DataDir --device $Device --seed $Seed --noise_type $nt --noise_rate $nr --epoch 30
}

Write-Host "=== Collecting Results ==="
python collect_results.py --result_dir ./results
