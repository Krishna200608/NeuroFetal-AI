
import subprocess
import time
import os
import signal
import sys

def test_app_startup():
    print("Testing app startup...")
    
    # Set CWD to Code/ directory
    cwd = os.path.join(os.getcwd(), 'Code')
    if not os.path.exists(cwd):
        print(f"Error: {cwd} does not exist.")
        cwd = os.getcwd() # Fallback

    # Run run_app.py
    # We use pwsh to avoid shell issues on Windows
    cmd = [sys.executable, "run_app.py"]
    
    process = subprocess.Popen(
        cmd, 
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    start_time = time.time()
    startup_success = False
    
    try:
        while time.time() - start_time < 15:
            # Check if process is still running
            if process.poll() is not None:
                print("Process terminated early.")
                break
                
            # Read output non-blocking? Hard in Python cross-platform without threads/async
            # We'll just read line by line with timeout logic if possible, 
            # but simpler approach:
            # Let it run for 10s, then kill and read output.
            time.sleep(1)
            
        print("Time's up. Terminating...")
        
    except KeyboardInterrupt:
        pass
    finally:
        # Kill process tree
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
        
        # Capture remaining output
        stdout, stderr = process.communicate()
        print("--- STDOUT ---")
        print(stdout)
        print("--- STDERR ---")
        print(stderr)
        
        if "DASHBOARD LIVE AT" in stdout or "LOCAL ADDRESS" in stdout:
            print("\nSUCCESS: Dashboard likely started.")
        else:
            print("\nFAILURE: Dashboard did not report live status.")

if __name__ == "__main__":
    test_app_startup()
