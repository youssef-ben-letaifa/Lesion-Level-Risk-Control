"""KiTS23 Segmentation Testing Suite Launcher.

Starts the interactive medical viewer web server and automatically opens
the browser at http://127.0.0.1:5000.
"""

import sys
import os
import time
import threading
import webbrowser

# Add interface folder to python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INTERFACE_DIR = os.path.join(ROOT_DIR, "interface")
sys.path.insert(0, INTERFACE_DIR)

import app


def open_browser():
    """Wait for server startup and open default browser."""
    time.sleep(1.5)
    print("\n[+] Opening web interface at: http://127.0.0.1:5000")
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    print("=" * 80)
    print("   KiTS23 SEGMENTATION TESTING SUITE & RISK CONTROL VIEWER")
    print("=" * 80)
    
    models = app.discover_models()
    print(f"\n[*] Discovered {len(models)} Trained .pth Checkpoints:")
    for m in models:
        def_tag = " [DEFAULT]" if m.get("is_default") else ""
        print(f"    - [{m['id']}] {m['filename']} ({m['size_mb']} MB){def_tag}")
        print(f"        {m['description']}")
        print(f"        Location: {m['rel_path']}")

    cases = app.discover_cases()
    pred_cases = [c for c in cases if c["has_prediction"]]
    print(f"\n[*] Discovered {len(cases)} Dataset Cases ({len(pred_cases)} with ready model predictions).")

    # Start browser in background thread
    threading.Thread(target=open_browser, daemon=True).start()

    print("\n[*] Starting Flask HTTP server on http://127.0.0.1:5000 ...")
    print("[*] Press Ctrl+C in this terminal window to stop the server.\n")
    
    app.app.run(host="127.0.0.1", port=5000, debug=False)

