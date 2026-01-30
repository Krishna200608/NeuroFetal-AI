import os
from pyngrok import ngrok
import subprocess
import time
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

def kill_ngrok_process():
    """Kills any existing ngrok.exe processes to avoid ERR_NGROK_334."""
    try:
        if os.name == 'nt':
            subprocess.run(["taskkill", "/IM", "ngrok.exe", "/F"], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
        else:
            subprocess.run(["pkill", "ngrok"], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
    except Exception:
        pass

def main():
    # 0. Clean up previous sessions
    kill_ngrok_process()

    # 1. Get Auth Token
    auth_token = os.getenv("NGROK_AUTH_TOKEN")
    if not auth_token:
        print("❌ Error: NGROK_AUTH_TOKEN not found in .env file.")
        print("Please create a .env file in the Code directory with NGROK_AUTH_TOKEN=your_token_here")
        return

    # 2. Authenticate ngrok
    print("🔐 Authenticating with ngrok...")
    ngrok.set_auth_token(auth_token)

    # 4. Start Streamlit App in Background
    print("🚀 Starting Streamlit Server...")
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
        print("❌ Streamlit failed to start.")
        print(process.stderr.read().decode())
        return

    # 4. Open ngrok Tunnel
    print("🌍 Opening public tunnel...")
    try:
        # Create a tunnel to port 8501
        public_url = ngrok.connect(8501).public_url
        print("\n" + "="*60)
        print(f"   🟢 DASHBOARD LIVE AT: {public_url}")
        print("="*60 + "\n")
        print("Press Ctrl+C to stop the server.")
        
        # Keep the script running
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            process.terminate()
            ngrok.kill()
            print("Done.")

    except Exception as e:
        print(f"❌ Error connecting ngrok: {e}")
        process.terminate()

if __name__ == "__main__":
    main()
