#!/usr/bin/env bash
# 重命名脚本：将旧的 ef-rl 文件名改为 e2c-rl
# 用于已经下载过旧文件名的情况

set -eo pipefail

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
RAW_RL_DIR="${DATA_DIR}/raw/rl"

echo -e "${BLUE}=== E2C RL 文件重命名脚本 ===${NC}"
echo "项目根目录: ${PROJECT_ROOT}"
echo "RL 数据目录: ${RAW_RL_DIR}"
echo ""

# 检查目录是否存在
if [ ! -d "${RAW_RL_DIR}" ]; then
    echo -e "${RED}错误: RL 数据目录不存在: ${RAW_RL_DIR}${NC}"
    echo "请先下载数据："
    echo "  bash scripts/download_datasets.sh --dataset rl"
    exit 1
fi

cd "${RAW_RL_DIR}"

# 重命名训练数据
if [ -f "ef-rl.parquet" ]; then
    if [ -f "e2c-rl.parquet" ]; then
        echo -e "${YELLOW}⚠️  e2c-rl.parquet 已存在，跳过重命名${NC}"
    else
        echo -e "${BLUE}重命名: ef-rl.parquet → e2c-rl.parquet${NC}"
        mv ef-rl.parquet e2c-rl.parquet
        echo -e "${GREEN}✓ 训练数据重命名成功${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  ef-rl.parquet 不存在${NC}"
fi

# 重命名验证数据
if [ -f "ef-rl-valid.parquet" ]; then
    if [ -f "e2c-rl-valid.parquet" ]; then
        echo -e "${YELLOW}⚠️  e2c-rl-valid.parquet 已存在，跳过重命名${NC}"
    else
        echo -e "${BLUE}重命名: ef-rl-valid.parquet → e2c-rl-valid.parquet${NC}"
        mv ef-rl-valid.parquet e2c-rl-valid.parquet
        echo -e "${GREEN}✓ 验证数据重命名成功${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  ef-rl-valid.parquet 不存在${NC}"
fi

echo ""
echo -e "${GREEN}=== 重命名完成 ===${NC}"
echo ""
echo "当前 RL 数据文件："
ls -lh "${RAW_RL_DIR}"/*.parquet 2>/dev/null || echo "  (无 parquet 文件)"
echo ""
echo -e "${BLUE}说明：${NC}"
echo "  • HuggingFace 上的文件名: ef-rl.parquet, ef-rl-valid.parquet"
echo "  • 本地统一使用: e2c-rl.parquet, e2c-rl-valid.parquet"
echo "  • 下载脚本已更新，新下载的文件会自动使用新名称"

