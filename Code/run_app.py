import os
from pyngrok import ngrok
import subprocess
import time
from dotenv import load_dotenv
import sys
import signal

# Load environment variables
load_dotenv()

def kill_process_on_port(port):
    """Kills the process listening on the specified port."""
    try:
        if os.name == 'nt':
            # Windows: Find PID using netstat
            cmd = f"netstat -ano | findstr :{port}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        if pid != '0':
                            print(f"   -> Killing process {pid} on port {port}...")
                            subprocess.run(f"taskkill /PID {pid} /F", shell=True, 
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # Linux/Mac: Find PID using lsof or fuser
            # Try lsof first
            cmd = f"lsof -t -i:{port}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.stdout:
                pid = result.stdout.strip()
                print(f"   -> Killing process {pid} on port {port}...")
                subprocess.run(f"kill -9 {pid}", shell=True)
    except Exception as e:
        print(f"Error cleaning up port {port}: {e}")

def main():
    # 0. Clean up previous sessions
    kill_process_on_port(8501)
    
    # Also kill explicit ngrok processes
    try:
        if os.name == 'nt':
            subprocess.run(["taskkill", "/IM", "ngrok.exe", "/F"], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(["pkill", "ngrok"], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except: pass

    # 1. Get Auth Token
    auth_token = os.getenv("NGROK_AUTH_TOKEN")
    if not auth_token:
        print("Error: NGROK_AUTH_TOKEN not found in .env file.")
        print("Please create a .env file in the Code directory with NGROK_AUTH_TOKEN=your_token_here")
        return

    # 2. Authenticate ngrok
    print("Authenticating with ngrok...")
    ngrok.set_auth_token(auth_token)

    # 4. Start Streamlit App in Background
    print("Starting Streamlit Server...")
    # Using subprocess to run the command: streamlit run scripts/app.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "scripts", "app.py")
    
    # Add Current Dir (Code/) to PYTHONPATH so scripts/app.py can import utils
    env = os.environ.copy()
    env["PYTHONPATH"] = current_dir + os.pathsep + env.get("PYTHONPATH", "")
    
    # Run streamlit
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", app_path, "--server.port", "8501", "--server.headless", "true"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )

    # Allow some time for streamlit to start
    time.sleep(3)

    if process.poll() is not None:
        print("Streamlit failed to start.")
        print(process.stderr.read().decode())
        return

    # 4. Open ngrok Tunnel (Optional)
    print("Attempting to open public tunnel...")
    public_url = None
    try:
        # Create a tunnel to port 8501
        public_url = ngrok.connect(8501).public_url
        print("\n" + "="*60)
        print(f"   DASHBOARD LIVE AT: {public_url}")
        print(f"   LOCAL ADDRESS:     http://localhost:8501")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"Ngrok Connection Failed: {e}")
        print("   (Continuing in Local-Only Mode)")
        print("\n" + "="*60)
        print(f"   LOCAL ADDRESS:     http://localhost:8501")
        print("="*60 + "\n")

    # --- Signal Handling for Graceful Shutdown ---
    
    def cleanup(signum=None, frame=None):
        print("\n🛑 Stopping NeuroFetal AI Dashboard...")
        
        # 1. Kill Streamlit
        if process:
            try:
                print("   -> Terminating Streamlit process...")
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception:
                pass

        # 2. Kill Ngrok
        try:
            print("   -> Closing Ngrok tunnels...")
            ngrok.kill()
            # Force kill system process just in case
            kill_process_on_port(8501)
        except Exception:
            pass
            
        print("✅ Shutdown Complete.")
        sys.exit(0)

    # Register signals
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("Press Ctrl+C to stop the server.")
    
    # Wait for process
    try:
        process.wait()
    except KeyboardInterrupt:
        cleanup()
    
if __name__ == "__main__":
    main()
