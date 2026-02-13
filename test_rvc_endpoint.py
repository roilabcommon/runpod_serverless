"""
RunPod Serverless Endpoint Tester — RVC Voice Conversion

Test your deployed RVC endpoint on RunPod.

Usage:
    # Using environment variables
    export RUNPOD_API_KEY="your_api_key"
    export RUNPOD_ENDPOINT_ID="your_endpoint_id"
    python test_rvc_endpoint.py --audio sample.wav

    # Using command line arguments
    python test_rvc_endpoint.py \
        --api-key your_api_key \
        --endpoint-id your_endpoint_id \
        --audio sample.wav \
        --character Poli \
        --language KR \
        --pitch 0
"""

import argparse
import base64
import os
import sys
import time

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[WARNING] python-dotenv not installed, .env file will be ignored")
    print("[TIP] Install it with: pip install python-dotenv")

try:
    import runpod
except ImportError:
    print("[ERROR] runpod package not installed")
    print("[TIP] Install it with: pip install runpod")
    sys.exit(1)


def load_audio_as_base64(audio_path: str) -> str:
    """Load audio file and convert to base64."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with open(audio_path, 'rb') as f:
        audio_data = f.read()

    return base64.b64encode(audio_data).decode('utf-8')


def save_audio_from_base64(base64_str: str, output_path: str):
    """Save base64 audio to file."""
    audio_data = base64.b64decode(base64_str)

    with open(output_path, 'wb') as f:
        f.write(audio_data)

    print(f"[OK] Audio saved to: {output_path}")


def test_rvc_endpoint(
    api_key: str,
    endpoint_id: str,
    audio_path: str,
    character: str = "Poli",
    language: str = "KR",
    pitch: int = 0,
    output_path: str = "rvc_output.wav",
):
    """Test RunPod serverless endpoint with RVC voice conversion."""

    print("=" * 60)
    print("RunPod Serverless RVC Endpoint Test")
    print("=" * 60)

    # Set API key
    runpod.api_key = api_key
    print(f"\n[KEY] API Key: {api_key[:10]}...{api_key[-4:]}")
    print(f"[ENDPOINT] Endpoint ID: {endpoint_id}")

    # Load audio
    print(f"\n[FILE] Loading audio from: {audio_path}")
    try:
        audio_base64 = load_audio_as_base64(audio_path)
        print(f"[OK] Audio loaded: {len(audio_base64)} bytes (base64)")
    except Exception as e:
        print(f"[ERROR] Failed to load audio: {e}")
        return False

    # Prepare input
    input_data = {
        "model_type": "rvc",
        "audio": audio_base64,
        "character": character,
        "language": language,
        "pitch_level": pitch,
    }

    print(f"\n[PARAMS] Request parameters:")
    print(f"  - Character: {character}")
    print(f"  - Language: {language}")
    print(f"  - Pitch level: {pitch}")

    # Create endpoint
    print(f"\n[CONNECT] Connecting to endpoint...")
    try:
        endpoint = runpod.Endpoint(endpoint_id)
    except Exception as e:
        print(f"[ERROR] Failed to connect to endpoint: {e}")
        return False

    # Run request
    print(f"[WAIT] Submitting job...")
    start_time = time.time()

    try:
        run_request = endpoint.run(input_data)
        job_id = run_request.job_id
        print(f"[OK] Job submitted: {job_id}")

        # Poll for result
        print(f"\n[WAIT] Waiting for result...")
        result = None
        last_status = None

        while True:
            status = run_request.status()

            if status != last_status:
                print(f"[STATUS] Status: {status}")
                last_status = status

            if status == "COMPLETED":
                result = run_request.output()
                break
            elif status == "FAILED":
                error = run_request.output()
                print(f"[ERROR] Job failed: {error}")
                return False
            elif status in ["CANCELLED", "TIMED_OUT"]:
                print(f"[ERROR] Job {status}")
                return False

            time.sleep(2)

        elapsed_time = time.time() - start_time
        print(f"\n[TIME] Total time: {elapsed_time:.2f} seconds")

        # Check result
        if not result:
            print("[ERROR] No result returned")
            return False

        if "error" in result:
            print(f"[ERROR] Error from endpoint: {result['error']}")
            return False

        # Display result info
        print(f"\n[OK] RVC conversion successful!")
        print(f"  - Model used: {result.get('model_used')}")
        print(f"  - Character: {result.get('character')}")
        print(f"  - Language: {result.get('language')}")
        print(f"  - Pitch level: {result.get('pitch_level')}")
        print(f"  - Audio size: {len(result.get('audio', ''))} bytes (base64)")

        # Save output
        print(f"\n[SAVE] Saving output to: {output_path}")
        try:
            save_audio_from_base64(result['audio'], output_path)
            print(f"\n[SUCCESS] Test completed successfully!")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save output: {e}")
            return False

    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test RunPod serverless RVC endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using environment variables
  export RUNPOD_API_KEY="your_api_key"
  export RUNPOD_ENDPOINT_ID="your_endpoint_id"
  python test_rvc_endpoint.py --audio sample.wav

  # Using command line arguments
  python test_rvc_endpoint.py \\
      --api-key YOUR_API_KEY \\
      --endpoint-id YOUR_ENDPOINT_ID \\
      --audio sample.wav \\
      --character Poli \\
      --language KR \\
      --pitch 0
        """
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("RUNPOD_API_KEY"),
        help="RunPod API key (or set RUNPOD_API_KEY env var)"
    )

    parser.add_argument(
        "--endpoint-id",
        type=str,
        default=os.environ.get("RUNPOD_ENDPOINT_ID"),
        help="RunPod endpoint ID (or set RUNPOD_ENDPOINT_ID env var)"
    )

    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to input audio file for voice conversion"
    )

    parser.add_argument(
        "--character",
        type=str,
        default="Poli",
        help="RVC character name (default: Poli)"
    )

    parser.add_argument(
        "--language",
        type=str,
        default="KR",
        help="Language folder (default: KR)"
    )

    parser.add_argument(
        "--pitch",
        type=int,
        default=0,
        help="Pitch level (default: 0)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="rvc_output.wav",
        help="Output audio file path (default: rvc_output.wav)"
    )

    args = parser.parse_args()

    # Validate required arguments
    if not args.api_key:
        print("[ERROR] Error: RunPod API key not provided")
        print("[TIP] Set RUNPOD_API_KEY environment variable or use --api-key")
        sys.exit(1)

    if not args.endpoint_id:
        print("[ERROR] Error: RunPod endpoint ID not provided")
        print("[TIP] Set RUNPOD_ENDPOINT_ID environment variable or use --endpoint-id")
        sys.exit(1)

    # Run test
    success = test_rvc_endpoint(
        api_key=args.api_key,
        endpoint_id=args.endpoint_id,
        audio_path=args.audio,
        character=args.character,
        language=args.language,
        pitch=args.pitch,
        output_path=args.output,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
