#!/usr/bin/env python3
"""
Demo Launcher - Helps start the distributed inference demo
"""
import subprocess
import sys
import time
import requests
import argparse

def check_dependencies():
    """Check if required packages are available"""
    required = ['iroh', 'torch', 'fastapi', 'uvicorn', 'requests', 'numpy']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print("Please install them first:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True

def wait_for_server(url="http://localhost:8000/health", timeout=30):
    """Wait for server to start"""
    print("⏳ Waiting for server to start...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print("✅ Server is ready!")
                return True
        except:
            pass
        time.sleep(1)
    
    print("❌ Server failed to start within timeout")
    return False

def get_ticket():
    """Get the Iroh ticket from the server"""
    try:
        response = requests.get("http://localhost:8000/ticket")
        if response.status_code == 200:
            return response.json()["ticket"]
    except:
        pass
    return None

def main():
    parser = argparse.ArgumentParser(description="Distributed Inference Demo Launcher")
    parser.add_argument("--mode", choices=["server", "peer", "client", "info"], 
                       default="info", help="What to run")
    parser.add_argument("--peer-id", help="Peer ID (for peer mode)")
    parser.add_argument("--layer-idx", type=int, help="Layer index (for peer mode)")
    parser.add_argument("--layer-type", choices=["embedding", "transformer", "output"],
                       help="Layer type (for peer mode)")
    
    args = parser.parse_args()
    
    print("🎯 Distributed Inference Demo Launcher")
    print("=" * 50)
    
    if not check_dependencies():
        return
    
    if args.mode == "server":
        print("🚀 Starting central server...")
        try:
            subprocess.run([sys.executable, "central_server_demo.py"], check=True)
        except KeyboardInterrupt:
            print("\n👋 Server stopped")
        except subprocess.CalledProcessError as e:
            print(f"❌ Server failed: {e}")
            
    elif args.mode == "peer":
        if not all([args.peer_id, args.layer_idx is not None, args.layer_type]):
            print("❌ Peer mode requires --peer-id, --layer-idx, and --layer-type")
            return
            
        # Get ticket from server
        print("🎟️ Getting ticket from server...")
        ticket = get_ticket()
        if not ticket:
            print("❌ Could not get ticket from server. Is the server running?")
            return
            
        print(f"🚀 Starting peer {args.peer_id} (layer {args.layer_idx}: {args.layer_type})...")
        try:
            subprocess.run([
                sys.executable, "peer_node_demo.py",
                "--peer-id", args.peer_id,
                "--layer-idx", str(args.layer_idx),
                "--layer-type", args.layer_type,
                "--ticket", ticket
            ], check=True)
        except KeyboardInterrupt:
            print(f"\n👋 Peer {args.peer_id} stopped")
        except subprocess.CalledProcessError as e:
            print(f"❌ Peer failed: {e}")
            
    elif args.mode == "client":
        print("🧪 Running test client...")
        if not wait_for_server():
            return
            
        try:
            subprocess.run([sys.executable, "test_client.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Client failed: {e}")
            
    else:  # info mode
        print("📖 Demo Instructions")
        print()
        print("This demo now uses Iroh Gossip for all control messages (triggers, completions) and Iroh Blobs for hidden state transfer. The document is no longer used for coordination.")
        print()
        print("🔧 Setup (run each in a separate terminal):")
        print()
        print("1. Start the central server:")
        print("   python run_demo.py --mode server")
        print()
        print("2. Start peer nodes:")
        print("   python run_demo.py --mode peer --peer-id peer_1 --layer-idx 0 --layer-type embedding")
        print("   python run_demo.py --mode peer --peer-id peer_2 --layer-idx 1 --layer-type transformer")
        print("   python run_demo.py --mode peer --peer-id peer_3 --layer-idx 2 --layer-type output")
        print()
        print("3. Run a test inference:")
        print("   python run_demo.py --mode client")
        print()
        print("🌐 Manual testing:")
        print("   Server UI: http://localhost:8000/docs")
        print("   Health: http://localhost:8000/health")
        print("   Ticket: http://localhost:8000/ticket")
        print()
        print("📚 For detailed architecture info, see:")
        print("   README_distributed_inference_demo.md")

if __name__ == "__main__":
    main() 