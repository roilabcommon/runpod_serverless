"""
RunPod Pod로 로컬 파일을 네트워크 볼륨에 업로드하는 스크립트

사전 준비:
  pip install paramiko requests

사용법:
  python upload_to_pod.py --api-key YOUR_API_KEY --pod-id YOUR_POD_ID

  또는 환경변수 설정 후:
  set RUNPOD_API_KEY=your_api_key
  python upload_to_pod.py --pod-id YOUR_POD_ID
"""

import os
import sys
import argparse
import requests
import paramiko
from pathlib import Path


GRAPHQL_URL = "https://api.runpod.io/graphql"

# 로컬 소스 → 네트워크 볼륨 목적지 매핑
UPLOAD_MAP = {
    "TTS/KR": "/runpod-volume/models/RVC/weights/KR",
}


def get_pod_ssh_info(api_key: str, pod_id: str) -> dict:
    """RunPod GraphQL API로 Pod SSH 접속 정보를 조회한다."""
    query = """
    query Pod($podId: String!) {
        pod(input: { podId: $podId }) {
            id
            name
            desiredStatus
            runtime {
                ports {
                    ip
                    isIpPublic
                    privatePort
                    publicPort
                    type
                }
            }
        }
    }
    """
    resp = requests.post(
        GRAPHQL_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"query": query, "variables": {"podId": pod_id}},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")

    pod = data["data"]["pod"]
    if not pod:
        raise RuntimeError(f"Pod '{pod_id}' not found")
    if pod["desiredStatus"] != "RUNNING":
        raise RuntimeError(f"Pod is not running (status: {pod['desiredStatus']})")

    # SSH 포트(22) 찾기
    ssh_port = None
    ssh_ip = None
    for port_info in pod["runtime"]["ports"]:
        if port_info["privatePort"] == 22 and port_info["isIpPublic"]:
            ssh_ip = port_info["ip"]
            ssh_port = port_info["publicPort"]
            break

    if not ssh_ip or not ssh_port:
        raise RuntimeError("SSH port not found. Pod에 SSH가 노출되어 있는지 확인하세요.")

    print(f"Pod: {pod['name']} ({pod_id})")
    print(f"SSH: {ssh_ip}:{ssh_port}")
    return {"ip": ssh_ip, "port": ssh_port}


def upload_files(ssh_info: dict, local_dir: str, remote_dir: str):
    """SFTP로 로컬 디렉토리의 파일들을 Pod에 업로드한다."""
    local_path = Path(local_dir)
    if not local_path.exists():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")

    files = [f for f in local_path.iterdir() if f.is_file()]
    if not files:
        print(f"No files in {local_dir}, skipping.")
        return

    total_size = sum(f.stat().st_size for f in files)
    print(f"\nUploading {len(files)} files ({total_size / 1024 / 1024:.1f} MB)")
    print(f"  From: {local_path.resolve()}")
    print(f"  To:   {remote_dir}")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(
            hostname=ssh_info["ip"],
            port=ssh_info["port"],
            username="root",
            # RunPod pods use key-based or password-less root access
            # If password is needed, add: password="your_password"
            timeout=15,
        )

        # 원격 디렉토리 생성
        ssh.exec_command(f"mkdir -p {remote_dir}")

        sftp = ssh.open_sftp()
        uploaded = 0

        for file in files:
            remote_file = f"{remote_dir}/{file.name}"
            file_size = file.stat().st_size
            uploaded += 1

            print(f"  [{uploaded}/{len(files)}] {file.name} ({file_size / 1024 / 1024:.1f} MB) ... ", end="", flush=True)
            sftp.put(str(file), remote_file)
            print("OK")

        sftp.close()

        # 업로드 결과 확인
        _, stdout, _ = ssh.exec_command(f"ls -lh {remote_dir}")
        print(f"\nRemote directory listing ({remote_dir}):")
        print(stdout.read().decode())

    finally:
        ssh.close()


def main():
    parser = argparse.ArgumentParser(description="Upload files to RunPod network volume via Pod SSH")
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"), help="RunPod API key")
    parser.add_argument("--pod-id", required=True, help="RunPod Pod ID")
    parser.add_argument("--local-dir", default=None, help="Override local source directory")
    parser.add_argument("--remote-dir", default=None, help="Override remote destination directory")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: RunPod API key required.")
        print("  --api-key YOUR_KEY  or  set RUNPOD_API_KEY=your_key")
        sys.exit(1)

    # SSH 접속 정보 조회
    ssh_info = get_pod_ssh_info(args.api_key, args.pod_id)

    # 업로드 실행
    if args.local_dir and args.remote_dir:
        upload_files(ssh_info, args.local_dir, args.remote_dir)
    else:
        script_dir = Path(__file__).resolve().parent
        for local_rel, remote in UPLOAD_MAP.items():
            local_abs = str(script_dir / local_rel)
            upload_files(ssh_info, local_abs, remote)

    print("Upload complete!")


if __name__ == "__main__":
    main()
