#!/bin/bash
# INR Digital Rock - 云端实验运行脚本
# 适用于 RTX 4090 (24GB)

# ============================================
# 配置区域 - 根据需要修改
# ============================================

# 数据路径（选择一个）
DATA_BEREA_ALTERED="DRP-265/Berea Altered_ mixed-wet/Berea_Altered_waterflooding_segmented/Berea_Altered_waterflooding_segmented.am"
DATA_BEREA_UNALTERED="DRP-265/Berea Unaltered_ water-wet/Berea_Unaltered_waterflooding_segmented/Berea_Unaltered_waterflooding_segmented.am"

# 默认使用 Altered 数据
DATA_PATH=$DATA_BEREA_ALTERED

# 输出目录
OUTPUT_DIR="output"

# 训练参数
EPOCHS=100

# ============================================
# 安装依赖
# ============================================
install_deps() {
    echo "=== Installing dependencies ==="
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install numpy scipy scikit-image matplotlib tqdm pillow
    echo "=== Dependencies installed ==="
}

# ============================================
# 快速测试（用小数据验证代码）
# ============================================
quick_test() {
    echo "=== Quick Test (256³ crop) ==="
    python run_experiments.py \
        --data "$DATA_PATH" \
        --output "${OUTPUT_DIR}/quick_test" \
        --epochs 20 \
        --crop_size 256 \
        --experiment compression
    echo "=== Quick test completed ==="
}

# ============================================
# 压缩实验（完整 1200³）
# ============================================
run_compression() {
    echo "=== Compression Experiment (Full 1200³) ==="
    python run_experiments.py \
        --data "$DATA_PATH" \
        --output "${OUTPUT_DIR}/compression_full" \
        --epochs $EPOCHS \
        --experiment compression
    echo "=== Compression experiment completed ==="
}

# ============================================
# 超分辨率实验（使用 600³ 子块作为 GT）
# ============================================
run_super_resolution() {
    echo "=== Super-Resolution Experiment (600³) ==="
    python run_experiments.py \
        --data "$DATA_PATH" \
        --output "${OUTPUT_DIR}/sr_full" \
        --epochs $EPOCHS \
        --crop_size 600 \
        --experiment sr
    echo "=== Super-resolution experiment completed ==="
}

# ============================================
# 运行所有实验
# ============================================
run_all() {
    echo "=== Running ALL Experiments ==="
    
    # 1. 压缩实验 (1200³)
    run_compression
    
    # 2. 超分辨率实验 (600³ 子块)
    run_super_resolution
    
    echo "=== ALL Experiments Completed ==="
    echo "Results saved to: ${OUTPUT_DIR}/"
}

# ============================================
# 两个数据集都跑
# ============================================
run_both_datasets() {
    echo "=== Running experiments on BOTH datasets ==="
    
    # Dataset 1: Berea Altered
    DATA_PATH=$DATA_BEREA_ALTERED
    OUTPUT_DIR="output/berea_altered"
    run_all
    
    # Dataset 2: Berea Unaltered
    DATA_PATH=$DATA_BEREA_UNALTERED
    OUTPUT_DIR="output/berea_unaltered"
    run_all
    
    echo "=== Both datasets completed ==="
}

# ============================================
# 使用说明
# ============================================
usage() {
    echo "Usage: bash run_cloud.sh [command]"
    echo ""
    echo "Commands:"
    echo "  install     - Install dependencies"
    echo "  quick       - Quick test with 256³ crop"
    echo "  compression - Run compression experiment (full 1200³)"
    echo "  sr          - Run super-resolution experiment (600³)"
    echo "  all         - Run all experiments"
    echo "  both        - Run all experiments on both datasets"
    echo ""
    echo "Example:"
    echo "  bash run_cloud.sh install"
    echo "  bash run_cloud.sh quick"
    echo "  bash run_cloud.sh all"
}

# ============================================
# 主入口
# ============================================
case "$1" in
    install)
        install_deps
        ;;
    quick)
        quick_test
        ;;
    compression)
        run_compression
        ;;
    sr)
        run_super_resolution
        ;;
    all)
        run_all
        ;;
    both)
        run_both_datasets
        ;;
    *)
        usage
        ;;
esac
