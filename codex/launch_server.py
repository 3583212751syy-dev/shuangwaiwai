#!/usr/bin/env python3
"""
Quick start: Launch the image-fission inference server.
Run this from: e:\Desktop\双接口\codex\
"""
import os
import sys
import subprocess
import time

def main():
    print("=" * 70)
    print("Image Fission Inference Server Launcher")
    print("=" * 70)
    print()
    
    # Check if model_worker exists
    model_worker_dir = os.path.join(os.getcwd(), 'model_worker')
    if not os.path.isdir(model_worker_dir):
        print("✗ ERROR: model_worker directory not found!")
        print(f"  Expected: {model_worker_dir}")
        sys.exit(1)
    
    print(f"✓ Found model_worker at: {model_worker_dir}")
    print()
    
    # Check for required packages
    print("Checking dependencies...")
    required = ['fastapi', 'uvicorn', 'PIL']
    missing = []
    for pkg in required:
        try:
            __import__(pkg.replace('PIL', 'PIL.Image'))
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} missing")
            missing.append(pkg)
    
    print()
    
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q'] + 
            ['fastapi', 'uvicorn', 'pillow'],
            cwd=os.getcwd()
        )
        print("✓ Installation complete")
        print()
    
    # Launch server
    print("=" * 70)
    print("Starting server on http://0.0.0.0:7860")
    print("=" * 70)
    print()
    print("Usage examples:")
    print()
    print("  curl -X POST 'http://localhost:7860/infer_enhanced' \\")
    print("    -F 'image=@your_image.png' \\")
    print("    -F 'mode=theme_background' \\")
    print("    -F 'theme=warm herbal' \\")
    print("    -F 'theme_strength=0.8' \\")
    print("    -F 'reference_range=0.4'")
    print()
    print("Press Ctrl+C to stop the server")
    print()
    print("=" * 70)
    print()
    
    # Run uvicorn
    os.chdir(model_worker_dir)
    subprocess.run(
        [sys.executable, '-m', 'uvicorn', 'worker_infer_enhanced:app', 
         '--host', '0.0.0.0', '--port', '7860', '--reload'],
        cwd=model_worker_dir
    )

if __name__ == '__main__':
    main()
