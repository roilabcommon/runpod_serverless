#!/bin/bash

set -e  # Exit on error

echo "================================="
echo "Cloning and Installing SparkTTS"
echo "================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TTS_DIR="$SCRIPT_DIR"

echo "TTS Directory: $TTS_DIR"

# Clone SparkTTS to temp directory
TEMP_DIR=$(mktemp -d)
echo "Temporary directory: $TEMP_DIR"

echo ""
echo "Cloning SparkTTS repository..."
if git clone --depth 1 https://github.com/SparkAudio/Spark-TTS.git "$TEMP_DIR/SparkTTS"; then
    echo "✓ SparkTTS cloned successfully"

    cd "$TEMP_DIR/SparkTTS"

    # Install requirements
    if [ -f "requirements.txt" ]; then
        echo ""
        echo "Installing SparkTTS requirements..."
        pip install --no-cache-dir -r requirements.txt || echo "⚠ Warning: Some requirements failed to install"
    fi

    # Install package
    echo ""
    echo "Installing SparkTTS package..."
    pip install --no-cache-dir -e . || echo "⚠ Warning: Package installation failed"

    # Copy only essential folders to TTS directory
    echo ""
    echo "Extracting SparkTTS essential folders..."

    # Copy sparktts folder
    if [ -d "$TEMP_DIR/SparkTTS/sparktts" ]; then
        echo "Copying sparktts..."
        rm -rf "$TTS_DIR/sparktts"
        cp -r "$TEMP_DIR/SparkTTS/sparktts" "$TTS_DIR/"
    else
        echo "⚠ Warning: sparktts folder not found"
    fi

    # Copy src folder
    if [ -d "$TEMP_DIR/SparkTTS/src" ]; then
        echo "Copying src..."
        rm -rf "$TTS_DIR/src"
        cp -r "$TEMP_DIR/SparkTTS/src" "$TTS_DIR/"
    else
        echo "⚠ Warning: src folder not found"
    fi

    # Copy cli folder
    if [ -d "$TEMP_DIR/SparkTTS/cli" ]; then
        echo "Copying cli..."
        rm -rf "$TTS_DIR/cli"
        cp -r "$TEMP_DIR/SparkTTS/cli" "$TTS_DIR/"
    else
        echo "⚠ Warning: cli folder not found"
    fi

    # Copy runtime folder
    if [ -d "$TEMP_DIR/SparkTTS/runtime" ]; then
        echo "Copying runtime..."
        rm -rf "$TTS_DIR/runtime"
        cp -r "$TEMP_DIR/SparkTTS/runtime" "$TTS_DIR/"
    else
        echo "⚠ Warning: runtime folder not found"
    fi

    # Cleanup
    cd "$TTS_DIR"
    rm -rf "$TEMP_DIR"

    echo ""
    echo "================================="
    echo "✓ SparkTTS installation completed!"
    echo "================================="
    echo "Extracted folders:"
    echo "  - sparktts: $TTS_DIR/sparktts"
    echo "  - src: $TTS_DIR/src"
    echo "  - cli: $TTS_DIR/cli"
    echo "  - runtime: $TTS_DIR/runtime"
else
    echo "✗ Failed to clone SparkTTS"
    rm -rf "$TEMP_DIR"
    exit 1
fi
