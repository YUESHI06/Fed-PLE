# CEGT_Fed

CEGT_Fed integrates the CEGT smart-contract graph model into the RESCUER-style
federated learning experiments.

## Implemented Requirements

- DAppSCAN processing from `DAppSCAN-source/SWCsource`, grouped by audit company.
- Four vulnerability tasks:
  - `integer_overflow`
  - `reentrancy`
  - `time_dependency`
  - `dos_failed_call`
- CEGT model replacement for the original RESCUER detector.
- Motivation experiments:
  - DAppSCAN train/test.
  - SmartBugs-Wild train, DAppSCAN test.
  - SmartBugs-Wild + Confidence Learning train, DAppSCAN test.
  - Per-party DAppSCAN local training with averaged metrics.
- Baselines:
  - RESCUER
  - FedAvg
  - CL
  - CLC
  - FedCorr
  - ARFL
- False-negative noise settings:
  - uniform `0.0`, `0.1`, `0.2`, `0.3`
  - asymmetric `0.1/0.2` and `0.1/0.3`
- Ablation experiments:
  - full model
  - without LCN
  - without warm-up
  - without orthogonal initialization
  - without Transformer

## Data Preparation

DAppSCAN is processed automatically when `./data/reentrancy` does not exist:

```powershell
python data_processing/dappscan_processor.py --dappscan_dir ../DAppSCAN-main/DAppSCAN-source --output_dir ./data --vul all
```

SmartBugs-Wild is optional but required for the real cross-dataset motivation
experiments. Put it at `../SmartBugs-Wild` or pass its path explicitly:

```powershell
python data_processing/smartbugs_processor.py --smartbugs_dir ../SmartBugs-Wild --output_dir ./data_smartbugs --vul all
```

If SmartBugs-Wild has not been processed, the code prints a warning and falls
back to a DAppSCAN company split so the experiment pipeline can still run.

## Run

Windows:

```powershell
Set-Location CEGT_Fed
.\run_all_experiments.ps1 -Device cuda:0
```

Linux/macOS:

```bash
cd CEGT_Fed
bash run_all_experiments.sh
```

Collect tables:

```powershell
python collect_results.py --result_dir ./results
```
