"""
RunPod Network Volume 파일 구조 확인 스크립트

이 스크립트를 RunPod Pod의 터미널에서 실행하여
Network Volume의 파일 구조를 확인할 수 있습니다.

사용법:
1. RunPod Pod 생성 (roi_ai_studio Volume 연결)
2. Web Terminal 접속
3. 이 스크립트를 복사하여 check_volume.py로 저장
4. python check_volume.py 실행
"""

import os
import json
from pathlib import Path

def get_dir_size(path):
    """디렉토리 크기 계산 (MB)"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_dir_size(entry.path)
    except PermissionError:
        pass
    return total / (1024 * 1024)  # MB로 변환

def scan_volume(volume_path="/workspace"):
    """Volume 구조 스캔"""

    print("=" * 60)
    print(f"Network Volume 구조 확인: {volume_path}")
    print("=" * 60)
    print()

    if not os.path.exists(volume_path):
        print(f"❌ Volume이 {volume_path}에 마운트되지 않았습니다.")
        print("💡 확인: df -h | grep volume")
        return

    # 루트 디렉토리 내용
    print(f"📁 Root Directory: {volume_path}")
    print("-" * 60)

    try:
        items = sorted(os.listdir(volume_path))
        for item in items:
            item_path = os.path.join(volume_path, item)
            if os.path.isdir(item_path):
                size = get_dir_size(item_path)
                print(f"  📂 {item}/ ({size:.2f} MB)")
            else:
                size = os.path.getsize(item_path) / (1024 * 1024)
                print(f"  📄 {item} ({size:.2f} MB)")
    except Exception as e:
        print(f"❌ 에러: {e}")

    print()

    # models 디렉토리 상세 확인
    models_path = os.path.join(volume_path, "models")
    if os.path.exists(models_path):
        print(f"📁 Models Directory: {models_path}")
        print("-" * 60)

        for model_dir in sorted(os.listdir(models_path)):
            model_path = os.path.join(models_path, model_dir)
            if not os.path.isdir(model_path):
                continue

            size = get_dir_size(model_path)
            print(f"\n  📦 {model_dir}/ ({size:.2f} MB)")

            # 파일 목록
            try:
                files = sorted(os.listdir(model_path))
                print(f"     파일 개수: {len(files)}")

                # 주요 파일만 표시
                important_files = [
                    f for f in files
                    if f.endswith(('.json', '.safetensors', '.bin', '.pt', '.pth', '.txt'))
                ]

                if important_files:
                    print("     주요 파일:")
                    for f in important_files[:10]:  # 최대 10개
                        file_path = os.path.join(model_path, f)
                        if os.path.isfile(file_path):
                            f_size = os.path.getsize(file_path) / (1024 * 1024)
                            print(f"       - {f} ({f_size:.2f} MB)")

                # 서브디렉토리 확인
                subdirs = [f for f in files if os.path.isdir(os.path.join(model_path, f))]
                if subdirs:
                    print(f"     서브디렉토리: {', '.join(subdirs[:5])}")

            except Exception as e:
                print(f"     ❌ 에러: {e}")
    else:
        print(f"⚠️  models 디렉토리가 없습니다: {models_path}")

    print()
    print("=" * 60)

    # 요약 JSON 출력
    summary = {
        "volume_path": volume_path,
        "exists": os.path.exists(volume_path),
        "total_size_mb": get_dir_size(volume_path) if os.path.exists(volume_path) else 0,
        "models": {}
    }

    if os.path.exists(models_path):
        for model_dir in os.listdir(models_path):
            model_path = os.path.join(models_path, model_dir)
            if os.path.isdir(model_path):
                summary["models"][model_dir] = {
                    "size_mb": get_dir_size(model_path),
                    "file_count": len(os.listdir(model_path)),
                    "has_config": os.path.exists(os.path.join(model_path, "config.json"))
                }

    print("\n📊 JSON Summary:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    # 여러 가능한 마운트 경로 확인
    possible_paths = [
        "/workspace",
        "/runpod-volume",
        os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume")
    ]

    for path in possible_paths:
        if os.path.exists(path):
            scan_volume(path)
            break
    else:
        print("❌ Volume을 찾을 수 없습니다.")
        print("💡 마운트된 볼륨 확인: df -h")
