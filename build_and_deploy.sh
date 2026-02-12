#!/bin/bash
# Docker 빌드 및 배포 스크립트
# Network Volume 지원 3가지 전략: volume-only, docker-embedded, hybrid

set -e  # 오류 발생 시 중단

# 기본값
DOCKER_USERNAME="${DOCKER_USERNAME:-your-username}"
IMAGE_NAME="runpod-tts-handler"
VERSION="latest"
STRATEGY="hybrid"
NO_PUSH=false
HELP=false

# 도움말 함수
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Docker 빌드 및 배포 스크립트 - RunPod Network Volume 지원

OPTIONS:
    --strategy <type>           배포 전략 선택 (기본값: hybrid)
                                  volume-only: Network Volume만 사용 (이미지 ~5-7GB)
                                  docker-embedded: Docker에 모델 포함 (이미지 ~20-25GB)
                                  hybrid: Docker + Volume 혼합 (권장, 이미지 ~20-25GB)

    --skip-models               --strategy volume-only의 단축키

    --docker-username <name>    Docker Hub 사용자명 (기본값: $DOCKER_USERNAME)

    --version <tag>             이미지 버전 태그 (기본값: latest)

    --no-push                   Docker Hub 푸시 스킵

    --help                      이 도움말 표시

EXAMPLES:
    # Volume-Only 전략 (개발용, 작은 이미지)
    $0 --strategy volume-only --docker-username myuser

    # Docker-Embedded 전략 (즉시 시작)
    $0 --strategy docker-embedded --docker-username myuser

    # Hybrid 전략 (프로덕션 권장)
    $0 --strategy hybrid --docker-username myuser --version v1.0.0

전략 비교:
    Volume-Only: 이미지 작음, 첫 시작 느림 (15-30분), 이후 빠름
    Docker-Embedded: 이미지 큼, 항상 즉시 시작
    Hybrid: 이미지 큼, 첫 시작 중간 속도 (5-10분), 이후 빠름

자세한 정보: NETWORK_STORAGE_GUIDE.md 참고
EOF
}

# 파라미터 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy)
            STRATEGY="$2"
            if [[ ! "$STRATEGY" =~ ^(volume-only|docker-embedded|hybrid)$ ]]; then
                echo "❌ Error: Invalid strategy '$STRATEGY'. Must be one of: volume-only, docker-embedded, hybrid"
                exit 1
            fi
            shift 2
            ;;
        --skip-models)
            STRATEGY="volume-only"
            shift
            ;;
        --docker-username)
            DOCKER_USERNAME="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --no-push)
            NO_PUSH=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# 환경 변수 검증
if [[ "$DOCKER_USERNAME" == "your-username" ]]; then
    echo "⚠️  Warning: Using default Docker username 'your-username'"
    echo "   Set DOCKER_USERNAME environment variable or use --docker-username flag"
    echo ""
fi

FULL_IMAGE="$DOCKER_USERNAME/$IMAGE_NAME:$VERSION"

# 전략별 빌드 인자 설정
case $STRATEGY in
    volume-only)
        SKIP_MODEL_DOWNLOAD="true"
        EXPECTED_SIZE="~5-7GB"
        ;;
    docker-embedded)
        SKIP_MODEL_DOWNLOAD="false"
        EXPECTED_SIZE="~20-25GB"
        ;;
    hybrid)
        SKIP_MODEL_DOWNLOAD="false"
        EXPECTED_SIZE="~20-25GB"
        ;;
esac

echo "===================================="
echo "Docker Build & Deploy Script"
echo "===================================="
echo "Strategy:        $STRATEGY"
echo "Image:           $FULL_IMAGE"
echo "Skip Models:     $SKIP_MODEL_DOWNLOAD"
echo "Expected Size:   $EXPECTED_SIZE"
echo ""

# 1. Docker 빌드
echo "📦 Building Docker image..."
echo "   This may take 15-45 minutes depending on strategy..."
echo ""

docker build \
    --build-arg SKIP_MODEL_DOWNLOAD=$SKIP_MODEL_DOWNLOAD \
    -t $IMAGE_NAME:$VERSION \
    .

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "✅ Build complete!"
echo ""

# 이미지 크기 표시
echo "📊 Image size information:"
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | grep -E "REPOSITORY|$IMAGE_NAME"
echo ""

# 실제 크기와 예상 크기 비교
ACTUAL_SIZE=$(docker images --format "{{.Size}}" $IMAGE_NAME:$VERSION)
echo "   Expected: $EXPECTED_SIZE"
echo "   Actual:   $ACTUAL_SIZE"
echo ""

# 2. 이미지 태그
echo "🏷️  Tagging image..."
docker tag $IMAGE_NAME:$VERSION $FULL_IMAGE
echo ""

# 3. Docker Hub 푸시 (선택)
if [ "$NO_PUSH" = false ]; then
    read -p "Push to Docker Hub? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📤 Pushing to Docker Hub..."
        echo "   This may take 10-60 minutes depending on image size..."
        docker push $FULL_IMAGE
        echo "✅ Push complete!"
    else
        echo "⏭️  Skipping push"
    fi
else
    echo "⏭️  Skipping push (--no-push flag)"
fi

echo ""
echo "===================================="
echo "✅ Done!"
echo "===================================="
echo ""

# 전략별 다음 단계 안내
echo "Next steps for $STRATEGY strategy:"
echo ""

case $STRATEGY in
    volume-only)
        cat << EOF
1. Create a RunPod Network Volume:
   - Go to RunPod Console > Serverless > Storage
   - Create new volume: name=roi_ai_studio, size=50GB
   - Note the mount path: /runpod-volume

2. Deploy to RunPod:
   - Container Image: $FULL_IMAGE
   - Network Volume: roi_ai_studio
   - Container Disk: 15GB (모델 미포함으로 작게 설정)
   - GPU: RTX 4090 or A100

3. First worker start:
   - Will download models from HuggingFace (15-30 minutes)
   - Subsequent workers will use cached models (fast)

4. Update .env with RUNPOD_ENDPOINT_ID

5. Test: python test_runpod_endpoint.py --text "Hello" --audio sample.wav --model spark

자세한 정보: NETWORK_STORAGE_GUIDE.md 참고
EOF
        ;;
    docker-embedded)
        cat << EOF
1. Deploy to RunPod (Network Volume 선택사항):
   - Container Image: $FULL_IMAGE
   - Container Disk: 30GB (모델 포함으로 크게 설정)
   - GPU: RTX 4090 or A100
   - Network Volume: (선택사항)

2. Workers will start immediately (모델 이미 포함됨)

3. Update .env with RUNPOD_ENDPOINT_ID

4. Test: python test_runpod_endpoint.py --text "Hello" --audio sample.wav --model spark

Note: Network Volume을 연결하면 향후 volume-only로 전환 가능
자세한 정보: NETWORK_STORAGE_GUIDE.md 참고
EOF
        ;;
    hybrid)
        cat << EOF
1. Create a RunPod Network Volume:
   - Go to RunPod Console > Serverless > Storage
   - Create new volume: name=roi_ai_studio, size=50GB
   - Note the mount path: /runpod-volume

2. Deploy to RunPod:
   - Container Image: $FULL_IMAGE
   - Network Volume: roi_ai_studio (반드시 연결!)
   - Container Disk: 30GB
   - GPU: RTX 4090 or A100

3. First worker start:
   - Will copy models from Docker → Volume (5-10 minutes)
   - Subsequent workers will use volume models (fast)

4. Update .env with RUNPOD_ENDPOINT_ID

5. Test: python test_runpod_endpoint.py --text "Hello" --audio sample.wav --model spark

이 전략은 프로덕션 환경에 권장됩니다.
자세한 정보: NETWORK_STORAGE_GUIDE.md 참고
EOF
        ;;
esac

echo ""
