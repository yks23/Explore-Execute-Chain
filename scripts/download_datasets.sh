#!/usr/bin/env bash
# E2C 数据集下载脚本
# 可以从 scripts 目录直接运行
# 用法：bash download_datasets.sh [--dataset sft|rl|eval|all] [--mirror]

set -eo pipefail

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get project root (parent of scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

# 默认参数
DATASET_TYPE="all"
RAW_DIR="${DATA_DIR}/raw"
EVAL_DIR="${DATA_DIR}/evaluation"

# HuggingFace 源配置
# 优先级：命令行参数 > 环境变量 HF_ENDPOINT > 默认值
USE_MIRROR=false
if [[ -n "${HF_ENDPOINT}" && "${HF_ENDPOINT}" == *"hf-mirror.com"* ]]; then
    USE_MIRROR=true
fi

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_TYPE="$2"
            shift 2
            ;;
        --mirror|--use-mirror)
            USE_MIRROR=true
            shift
            ;;
        --no-mirror)
            USE_MIRROR=false
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --dataset TYPE    指定下载类型 (sft|rl|eval|all), 默认: all"
            echo "  --mirror          使用镜像源 (hf-mirror.com)"
            echo "  --no-mirror       使用官方源 (huggingface.co, 默认)"
            echo "  -h, --help        显示帮助信息"
            echo ""
            echo "环境变量:"
            echo "  HF_ENDPOINT       设置为包含 'hf-mirror.com' 的 URL 将自动使用镜像源"
            echo ""
            echo "示例:"
            echo "  $0 --dataset eval --mirror    # 使用镜像源下载评估数据"
            echo "  HF_ENDPOINT=https://hf-mirror.com $0  # 通过环境变量使用镜像源"
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知参数 $1${NC}"
            exit 1
            ;;
    esac
done

# 设置 HuggingFace 基础 URL
if [ "$USE_MIRROR" = true ]; then
    HF_BASE_URL="https://hf-mirror.com"
    SOURCE_NAME="镜像源 (hf-mirror.com)"
else
    HF_BASE_URL="https://huggingface.co"
    SOURCE_NAME="官方源 (huggingface.co)"
fi

echo -e "${GREEN}=== E2C 数据集下载脚本 ===${NC}"
echo "项目根目录: ${PROJECT_ROOT}"
echo "数据目录: ${DATA_DIR}"
echo "数据类型: ${DATASET_TYPE}"
echo -e "下载源: ${BLUE}${SOURCE_NAME}${NC}"
echo ""

# 创建目录
mkdir -p "${RAW_DIR}/sft"
mkdir -p "${RAW_DIR}/rl"
mkdir -p "${EVAL_DIR}"

# 下载 SFT 数据
download_sft_data() {
    echo -e "${YELLOW}[SFT] 开始下载 SFT 训练数据...${NC}"
    
    # 从 Hugging Face 下载 E2C-SFT 数据集（在根目录）
    local SFT_URL="${HF_BASE_URL}/datasets/anomyous-author/Explore-Execute-Chain-Datasets/resolve/main/e2c-sft.parquet"
    local SFT_FILE="${RAW_DIR}/sft/e2c-sft.parquet"
    
    if [ -f "${SFT_FILE}" ]; then
        echo -e "${YELLOW}文件已存在: ${SFT_FILE}${NC}"
        read -p "是否覆盖? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "跳过下载"
            return 0
        fi
    fi
    
    echo "下载到: ${SFT_FILE}"
    wget -O "${SFT_FILE}" "${SFT_URL}" || {
        echo -e "${RED}下载失败，尝试使用 curl...${NC}"
        curl -L -o "${SFT_FILE}" "${SFT_URL}" || {
            echo -e "${RED}下载失败！请检查网络连接${NC}"
            return 1
        }
    }
    
    echo -e "${GREEN}✓ SFT 数据下载完成 (77.7 MB)${NC}"
}

# 下载 RL 数据
download_rl_data() {
    echo -e "${YELLOW}[RL] 开始下载 RL 训练数据...${NC}"
    
    # 从 Hugging Face 下载 E2C-RL 数据集（在根目录，下载后重命名）
    # HuggingFace 上的文件名: ef-rl.parquet, ef-rl-valid.parquet
    # 本地保存名称: e2c-rl.parquet, e2c-rl-valid.parquet
    local RL_TRAIN_URL="${HF_BASE_URL}/datasets/anomyous-author/Explore-Execute-Chain-Datasets/resolve/main/ef-rl.parquet"
    local RL_VALID_URL="${HF_BASE_URL}/datasets/anomyous-author/Explore-Execute-Chain-Datasets/resolve/main/ef-rl-valid.parquet"
    local RL_TRAIN_FILE="${RAW_DIR}/rl/e2c-rl.parquet"
    local RL_VALID_FILE="${RAW_DIR}/rl/e2c-rl-valid.parquet"
    
    # 下载训练数据
    if [ -f "${RL_TRAIN_FILE}" ]; then
        echo -e "${YELLOW}训练数据已存在: ${RL_TRAIN_FILE}${NC}"
    else
        echo "下载 RL 训练数据 (ef-rl.parquet → e2c-rl.parquet)..."
        wget -O "${RL_TRAIN_FILE}" "${RL_TRAIN_URL}" || {
            echo -e "${RED}下载失败，尝试使用 curl...${NC}"
            curl -L -o "${RL_TRAIN_FILE}" "${RL_TRAIN_URL}" || {
                echo -e "${RED}RL 训练数据下载失败！${NC}"
                return 1
            }
        }
        echo -e "${GREEN}✓ RL 训练数据下载完成: ${RL_TRAIN_FILE} (19.4 MB)${NC}"
    fi
    
    # 下载验证数据
    if [ -f "${RL_VALID_FILE}" ]; then
        echo -e "${YELLOW}验证数据已存在: ${RL_VALID_FILE}${NC}"
    else
        echo "下载 RL 验证数据 (ef-rl-valid.parquet → e2c-rl-valid.parquet)..."
        wget -O "${RL_VALID_FILE}" "${RL_VALID_URL}" || {
            echo -e "${RED}下载失败，尝试使用 curl...${NC}"
            curl -L -o "${RL_VALID_FILE}" "${RL_VALID_URL}" || {
                echo -e "${RED}RL 验证数据下载失败！${NC}"
                return 1
            }
        }
        echo -e "${GREEN}✓ RL 验证数据下载完成: ${RL_VALID_FILE} (706 KB)${NC}"
    fi
    
    echo -e "${GREEN}✓ RL 数据全部下载完成 (重命名为 e2c-rl*)${NC}"
}

# 下载评估数据集
download_eval_data() {
    echo -e "${YELLOW}[EVAL] 开始下载评估数据集...${NC}"
    
    # 所有评估数据集（在 HuggingFace 的 evaluation/ 目录下）
    local EVAL_DATASETS=(
        # 数学类数据集
        "aime24"
        "aime25"
        "amc23"
        "gsm8k"
        "math-algebra"
        "math500"
        "minerva"
        "olympiad_bench"
        # 医学类数据集
        "anatomy"
        "clinical_knowledge"
        "college_biology"
        "college_medicine"
        "medical_genetics"
        "medmcqa"
        "medqa"
        "professional_medicine"
    )
    
    mkdir -p "${EVAL_DIR}"
    
    local base_url="${HF_BASE_URL}/datasets/anomyous-author/Explore-Execute-Chain-Datasets/resolve/main/evaluation"
    
    echo "下载评估数据集..."
    for dataset_name in "${EVAL_DATASETS[@]}"; do
        local url="${base_url}/${dataset_name}.parquet"
        local output_file="${EVAL_DIR}/${dataset_name}.parquet"
        
        if [ -f "${output_file}" ]; then
            echo "  - ${dataset_name}: 已存在，跳过"
            continue
        fi
        
        echo "  - 下载 ${dataset_name}..."
        wget -q -O "${output_file}" "${url}" || {
            curl -s -L -o "${output_file}" "${url}" || {
                echo -e "${RED}    下载失败: ${dataset_name}${NC}"
                continue
            }
        }
        echo -e "${GREEN}    ✓ ${dataset_name} 下载完成${NC}"
    done
    
    echo -e "${GREEN}✓ 评估数据下载完成${NC}"
}

# 主逻辑
case "${DATASET_TYPE}" in
    sft)
        download_sft_data
        ;;
    rl)
        download_rl_data
        ;;
    eval)
        download_eval_data
        ;;
    all)
        download_sft_data
        download_rl_data
        download_eval_data
        ;;
    *)
        echo -e "${RED}错误: 未知的数据集类型 '${DATASET_TYPE}'${NC}"
        echo "支持的类型: sft, rl, eval, all"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}=== 下载完成 ===${NC}"
echo "数据位置:"
echo "  - SFT 原始数据: ${RAW_DIR}/sft/"
echo "  - RL 原始数据: ${RAW_DIR}/rl/"
echo "  - 评估数据: ${EVAL_DIR}/"
echo ""
echo "下一步:"
echo "  1. 准备所有数据: bash scripts/prepare_all_data.sh"
echo "  2. 运行评估: bash scripts/eval.sh"

