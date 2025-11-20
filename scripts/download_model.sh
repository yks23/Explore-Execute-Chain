#!/usr/bin/env bash
# E2C Model Download Script
# Download E2C models using HuggingFace (with mirror support)
# Usage: bash download_model.sh [--model MODEL_NAME] [--subfolder SUBFOLDER] [--mirror]

set -eo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Defaults
MODEL_NAME="anomyous-author/Explore-Execute-Chain"
SUBFOLDER="4B-Final"
USE_MIRROR=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --subfolder)
            SUBFOLDER="$2"
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
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model MODEL       Model name (default: anomyous-author/Explore-Execute-Chain)"
            echo "  --subfolder NAME    Subfolder (default: 4B-Final, options: 4B-Final, 8B-Final)"
            echo "  --mirror            Use mirror source (hf-mirror.com)"
            echo "  --no-mirror         Use official source (huggingface.co, default)"
            echo "  -h, --help          Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --subfolder 8B-Final --mirror    # Download 8B model using mirror"
            echo "  $0 --subfolder 4B-Final              # Download 4B model from official source"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown argument $1${NC}"
            exit 1
            ;;
    esac
done

# Set HuggingFace endpoint
if [ "$USE_MIRROR" = true ]; then
    export HF_ENDPOINT=https://hf-mirror.com
    SOURCE_NAME="镜像源 (hf-mirror.com)"
else
    export HF_ENDPOINT=https://huggingface.co
    SOURCE_NAME="官方源 (huggingface.co)"
fi

echo -e "${GREEN}=== E2C Model Download ===${NC}"
echo "Model: ${MODEL_NAME}"
echo "Subfolder: ${SUBFOLDER}"
echo -e "Source: ${BLUE}${SOURCE_NAME}${NC}"
echo ""

# Check Python and transformers
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

if ! python3 -c "import transformers" 2>/dev/null; then
    echo -e "${YELLOW}Warning: transformers not installed. Installing...${NC}"
    pip install transformers torch -q
fi

echo "Downloading model..."
echo ""

# Download model using Python
python3 << EOF
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "${MODEL_NAME}"
subfolder = "${SUBFOLDER}"
use_mirror = ${USE_MIRROR}

if use_mirror:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    print(f"Using mirror: {os.environ.get('HF_ENDPOINT')}")

print(f"Downloading model: {model_name}/{subfolder}")
print("This may take a while depending on your internet connection...")
print("")

try:
    # Download tokenizer first (smaller, faster)
    print("1. Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        subfolder=subfolder,
        trust_remote_code=True
    )
    print(f"   ✓ Tokenizer downloaded (vocab size: {len(tokenizer)})")
    
    # Download model
    print("2. Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        subfolder=subfolder,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    print(f"   ✓ Model downloaded")
    print("")
    print("✓ Model download complete!")
    print(f"  Model is cached at: {model.config.name_or_path}")
    
except Exception as e:
    print(f"✗ Download failed: {e}")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=== Download Complete ===${NC}"
    echo ""
    echo "Model is now cached and ready to use."
    echo "You can use it in your code with:"
    echo "  from transformers import AutoModelForCausalLM, AutoTokenizer"
    echo "  model = AutoModelForCausalLM.from_pretrained('${MODEL_NAME}', subfolder='${SUBFOLDER}')"
else
    echo -e "${RED}Download failed!${NC}"
    exit 1
fi

