#!/bin/bash
#SBATCH --job-name=zkllm-tests
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/sharefs/user50/zk/logs/tests-%j.out
#SBATCH --error=/mnt/sharefs/user50/zk/logs/tests-%j.err

mkdir -p /mnt/sharefs/user50/zk/logs
cd /mnt/sharefs/user50/zk/zkllm-ccs2024

PASS=0
FAIL=0

run_test() {
    echo "=== $1 ==="
    ./$1
    if [ $? -eq 0 ]; then
        echo "PASS"
        PASS=$((PASS+1))
    else
        echo "FAIL"
        FAIL=$((FAIL+1))
    fi
    echo ""
}

run_test test_zkargmax
run_test test_zklog
run_test test_zknormalcdf
run_test test_zkentropy

echo "=== Results: $PASS passed, $FAIL failed ==="
