#!/bin/bash

set -e  # Exit on error

echo "================================="
echo "Building RunPod Serverless Docker Image"
echo "================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Warning: docker-compose is not installed"
    echo "Using docker build instead..."
    USE_COMPOSE=false
else
    USE_COMPOSE=true
fi

# Build options
IMAGE_NAME="runpod-serverless:latest"
BUILD_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            BUILD_ARGS="$BUILD_ARGS --no-cache"
            shift
            ;;
        --image-name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--no-cache] [--image-name <name>]"
            exit 1
            ;;
    esac
done

# Build the image
if [ "$USE_COMPOSE" = true ]; then
    echo "Building with docker-compose..."
    docker-compose build $BUILD_ARGS
else
    echo "Building with docker..."
    docker build $BUILD_ARGS -t $IMAGE_NAME .
fi

echo ""
echo "================================="
echo "Build completed successfully!"
echo "================================="
echo ""
echo "To run the container:"
echo "  docker-compose up -d"
echo "  or"
echo "  docker run -it --gpus all -v \$(pwd)/TTS/pretrained_models:/app/TTS/pretrained_models -v \$(pwd)/RVC/weights:/app/RVC/weights $IMAGE_NAME"
echo ""
