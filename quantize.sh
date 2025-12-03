#!/bin/bash

# ================= USER CONFIGURATION =================
# 1. Model ID
MODEL_ID="Qwen/Qwen3-Embedding-0.6B"

# 2. Directories
FINAL_MODELS_DIR="models"           # Final destination
RAW_DOWNLOAD_DIR="raw_model_qwen3"  # Temp download folder

# 3. Output Filenames
FP16_NAME="qwen3-embed-f16.gguf"
FINAL_NAME_Q4="qwen3-embed-0.6b-q4_k_m.gguf"
FINAL_FILE_PATH="$FINAL_MODELS_DIR/$FINAL_NAME_Q4"

# 4. Files to Download via Curl
DOWNLOAD_FILES=(
    "config.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "vocab.json"
    "merges.txt"
    "model.safetensors"
)
# ======================================================

set -e # Stop execution on any error

# --- Step 0: Install Dependencies ---
echo ">>> [0/7] Checking System Dependencies..."

# 1. Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python3 is not installed. Please install it first."
    exit 1
fi

# 2. Check for Curl
if ! command -v curl &> /dev/null; then
    echo "   - Curl not found. Installing..."
    if command -v apt-get &> /dev/null; then sudo apt-get install -y curl
    elif command -v dnf &> /dev/null; then sudo dnf install -y curl
    elif command -v apk &> /dev/null; then sudo apk add curl
    elif command -v brew &> /dev/null; then brew install curl
    elif command -v pacman &> /dev/null; then sudo pacman -S --noconfirm curl
    else
        echo "❌ Error: Curl missing. Please install manually."
        exit 1
    fi
fi

# 3. Install CMake via Python
if ! command -v cmake &> /dev/null; then
    echo "   - CMake not found. Installing via pip..."
    pip install cmake
    export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v cmake &> /dev/null; then
    echo "❌ Error: CMake installed but not found in PATH."
    echo "   Try running: export PATH=\$HOME/.local/bin:\$PATH"
    exit 1
fi
echo "   - CMake Version: $(cmake --version | head -n 1)"

# --- Step 1: Setup Directories ---
echo ">>> [1/7] Setting up directories..."
mkdir -p "$FINAL_MODELS_DIR"
mkdir -p "$RAW_DOWNLOAD_DIR"

# --- Step 1.5: Smart State Detection ---
SKIP_HEAVY_LIFTING=false # Download, Convert, Quantize
DO_MOVE_AND_CLEANUP=true # Move final file, delete temp files

if [ -f "$FINAL_FILE_PATH" ]; then
    # Case A: Model exists in the final 'models/' folder.
    echo "✅ Found finished model in: $FINAL_FILE_PATH"
    echo "   Skipping Download, Conversion, Quantization, and Move."
    SKIP_HEAVY_LIFTING=true
    DO_MOVE_AND_CLEANUP=false

elif [ -f "$FINAL_NAME_Q4" ]; then
    # Case B: Model was quantized but script crashed before moving.
    echo "⚠️  Found orphaned quantized model in root directory ($FINAL_NAME_Q4)."
    echo "   Skipping Download and Quantization. Will resume at Move & Cleanup."
    SKIP_HEAVY_LIFTING=true
    DO_MOVE_AND_CLEANUP=true
fi

# --- Step 2: Robust Download (Retry Logic) ---
if [ "$SKIP_HEAVY_LIFTING" = false ]; then
    echo ">>> [2/7] Downloading $MODEL_ID via Curl (Resumable & Robust)..."
    
    BASE_URL="https://huggingface.co/$MODEL_ID/resolve/main"
    MAX_RETRIES=5
    
    for file in "${DOWNLOAD_FILES[@]}"; do
        OUTPUT_FILE="$RAW_DOWNLOAD_DIR/$file"
        
        if [ -f "$OUTPUT_FILE" ]; then
            echo "   - $file exists. Assuming complete (or will resume)."
        fi

        echo "   - Downloading $file..."
        
        ATTEMPT=1
        SUCCESS=false
        
        while [ $ATTEMPT -le $MAX_RETRIES ]; do
            if curl -L -f -C - \
                --retry 3 \
                --connect-timeout 20 \
                --speed-time 30 --speed-limit 100 \
                -o "$OUTPUT_FILE" "$BASE_URL/$file"; then
                
                SUCCESS=true
                break
            else
                echo "     ⚠️ Download failed/stuck on Attempt $ATTEMPT/$MAX_RETRIES."
                echo "     ⏳ Waiting 5 seconds before retrying..."
                sleep 5
                ((ATTEMPT++))
            fi
        done
        
        if [ "$SUCCESS" = false ]; then
            echo "❌ Error: Failed to download $file after $MAX_RETRIES attempts."
            exit 1
        fi
    done
    echo "   - All files downloaded successfully."
else
    echo ">>> [2/7] Download skipped (Model exists or ready to move)."
fi

# --- Step 3: Download & Build llama.cpp (NO GIT) ---
echo ">>> [3/7] Checking/Building llama.cpp..."

if [ ! -d "llama.cpp" ]; then
    echo "   - Downloading llama.cpp source code..."
    curl -L -o llama.tar.gz https://github.com/ggerganov/llama.cpp/archive/refs/heads/master.tar.gz
    echo "   - Extracting source..."
    tar -xf llama.tar.gz
    mv llama.cpp-master llama.cpp
    rm llama.tar.gz
fi

cd llama.cpp
pip install -r requirements.txt > /dev/null 2>&1

if [ ! -f "build/bin/llama-quantize" ] || [ ! -f "build/bin/llama-cli" ]; then
    echo "   - Building binaries..."
    rm -rf build
    cmake -B build -DLLAMA_CURL=OFF
    cmake --build build --config Release -j$(nproc) --target llama-quantize llama-cli
else
    echo "   - Binaries already built."
fi
cd ..

# --- Step 4: Convert to FP16 ---
if [ "$SKIP_HEAVY_LIFTING" = false ]; then
    echo ">>> [4/7] Converting HF model to FP16..."

    if [ -f "$FP16_NAME" ]; then
        echo "   - FP16 file already exists, skipping conversion."
    else
        if [ ! -f "$RAW_DOWNLOAD_DIR/model.safetensors" ]; then
             echo "❌ Error: model.safetensors not found."
             exit 1
        fi

        python3 llama.cpp/convert_hf_to_gguf.py "$RAW_DOWNLOAD_DIR" \
            --outfile "$FP16_NAME" \
            --outtype f16
    fi
else
    echo ">>> [4/7] Conversion skipped."
fi

# --- Step 5: Quantize (Q4 Only) ---
QUANTIZE_BIN="./llama.cpp/build/bin/llama-quantize"

if [ "$SKIP_HEAVY_LIFTING" = false ]; then
    echo ">>> [5/7] Generating Quantized Model (Q4_K_M)..."

    if [ -f "$FINAL_NAME_Q4" ]; then rm "$FINAL_NAME_Q4"; fi
    "$QUANTIZE_BIN" "$FP16_NAME" "$FINAL_NAME_Q4" q4_k_m
else
    echo ">>> [5/7] Quantization skipped."
fi

# --- Step 6: Move & Cleanup (With Permission Fix) ---
if [ "$DO_MOVE_AND_CLEANUP" = true ]; then
    echo ">>> [6/7] Organizing and Cleaning up..."

    # 1. Move the model
    if [ -f "$FINAL_NAME_Q4" ]; then
        echo "   - Moving '$FINAL_NAME_Q4' to '$FINAL_MODELS_DIR/'..."
        
        # Try standard move, if fail, try SUDO
        if ! mv "$FINAL_NAME_Q4" "$FINAL_MODELS_DIR/" 2>/dev/null; then
            echo "   ⚠️  Permission Denied. Attempting to move with 'sudo'..."
            sudo mv "$FINAL_NAME_Q4" "$FINAL_MODELS_DIR/"
            
            # Fix ownership back to the current user so you can use it later
            echo "   - Fixing file ownership..."
            sudo chown $(id -u):$(id -g) "$FINAL_FILE_PATH"
        fi
    else
        echo "   ⚠️  Warning: Could not find '$FINAL_NAME_Q4' to move."
    fi

    # 2. Cleanup Raw Directory
    if [ -d "$RAW_DOWNLOAD_DIR" ]; then
        echo "   - Removing temporary raw download folder..."
        rm -rf "$RAW_DOWNLOAD_DIR"
    fi

    # 3. Cleanup FP16 file
    if [ -f "$FP16_NAME" ]; then
        echo "   - Removing huge intermediate FP16 file..."
        rm "$FP16_NAME"
    fi
else
    echo ">>> [6/7] Cleanup skipped (Model already in destination)."
fi

echo "---------------------------------------------------"
echo "✅ SUCCESS!"
echo "1. $PWD/$FINAL_FILE_PATH"
echo "---------------------------------------------------"

# --- Step 7: Test ---
echo ">>> [7/7] Testing Q4 Model..."
./llama.cpp/build/bin/llama-cli \
    -m "$FINAL_FILE_PATH" \
    -p "Hello Qwen" \
    --embedding \
    --no-display-prompt \
    2>/dev/null | head -n 5

echo "..."