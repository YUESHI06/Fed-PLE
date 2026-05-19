#!/bin/bash
# ====================================================================
# run_all_experiments.sh - Master script for all experiments
# ====================================================================
# Usage:
#   1. First process data:  python data_processing/dappscan_processor.py
#   2. Then run experiments: bash run_all_experiments.sh
# ====================================================================

set -e
DATA_DIR="./data"
SMARTBUGS_RAW="../SmartBugs-Wild"
SMARTBUGS_DATA="./data_smartbugs"
DEVICE="cuda:0"
SEED=42

VULS=("reentrancy" "integer_overflow" "time_dependency" "dos_failed_call")

# ====================================================================
# Step 0: Data Processing (if not done yet)
# ====================================================================
if [ ! -d "$DATA_DIR/reentrancy" ]; then
    echo "=== Processing DAppSCAN dataset ==="
    python data_processing/dappscan_processor.py \
        --dappscan_dir ../DAppSCAN-main/DAppSCAN-source \
        --output_dir $DATA_DIR \
        --vul all \
        --seed $SEED
fi

if [ -d "$SMARTBUGS_RAW" ] && [ ! -d "$SMARTBUGS_DATA/reentrancy" ]; then
    echo "=== Processing SmartBugs-Wild dataset ==="
    python data_processing/smartbugs_processor.py \
        --smartbugs_dir "$SMARTBUGS_RAW" \
        --output_dir "$SMARTBUGS_DATA" \
        --vul all \
        --seed $SEED
fi

# ====================================================================
# Task 3: Motivation Experiments
# ====================================================================
echo ""
echo "============================================================"
echo "Task 3: Motivation Experiments"
echo "============================================================"
python non_Fed_Train.py \
    --data_dir $DATA_DIR \
    --smartbugs_data_dir $SMARTBUGS_DATA \
    --device $DEVICE \
    --seed $SEED

# ====================================================================
# Task 4: Baseline Comparison Experiments
# ====================================================================
echo ""
echo "============================================================"
echo "Task 4: Baseline Comparison"
echo "============================================================"

METHODS=("Fed_RESCUER" "Fed_FedAvg" "Fed_CL" "Fed_CLC" "Fed_FedCorr" "Fed_ARFL")
NOISE_RATES=(0.0 0.1 0.2 0.3)

for vul in "${VULS[@]}"; do
    echo ""
    echo "=== Vulnerability: $vul ==="

    for method in "${METHODS[@]}"; do
        # Pure (no noise)
        echo "  --- $method, noise=pure ---"
        python ${method}.py \
            --vul $vul \
            --data_dir $DATA_DIR \
            --device $DEVICE \
            --seed $SEED \
            --noise_type pure \
            --noise_rate 0.0 \
            --epoch 30

        # False-negative noise at various rates
        for nr in 0.1 0.2 0.3; do
            echo "  --- $method, fn_noise=$nr ---"
            python ${method}.py \
                --vul $vul \
                --data_dir $DATA_DIR \
                --device $DEVICE \
                --seed $SEED \
                --noise_type fn_noise \
                --noise_rate $nr \
                --epoch 30
        done
    done

    # Asymmetric noise: different noise rates per client
    echo "  --- Asymmetric noise experiments ---"
    for method in "${METHODS[@]}"; do
        # 10%/20% split
        echo "  --- $method, diff_noise 10%/20% ---"
        python ${method}.py \
            --vul $vul \
            --data_dir $DATA_DIR \
            --device $DEVICE \
            --seed $SEED \
            --noise_type diff_noise \
            --noise_rates 0.1 0.2 0.1 0.2 0.1 0.2 0.1 0.2 \
            --epoch 30

        # 10%/30% split
        echo "  --- $method, diff_noise 10%/30% ---"
        python ${method}.py \
            --vul $vul \
            --data_dir $DATA_DIR \
            --device $DEVICE \
            --seed $SEED \
            --noise_type diff_noise \
            --noise_rates 0.1 0.3 0.1 0.3 0.1 0.3 0.1 0.3 \
            --epoch 30
    done
done

# ====================================================================
# Task 5: Ablation Experiments
# ====================================================================
echo ""
echo "============================================================"
echo "Task 5: Ablation Experiments"
echo "============================================================"

for nr in 0.0 0.1 0.2 0.3; do
    echo "  --- Ablation with noise_rate=$nr ---"
    if [ "$nr" = "0.0" ]; then
        NT="pure"
    else
        NT="fn_noise"
    fi
    python Ablation.py \
        --data_dir $DATA_DIR \
        --device $DEVICE \
        --seed $SEED \
        --noise_type $NT \
        --noise_rate $nr \
        --epoch 30
done

echo ""
echo "============================================================"
echo "All experiments completed!"
echo "Results saved in ./results/"
echo "============================================================"
