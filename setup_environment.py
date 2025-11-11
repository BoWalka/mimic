#!/usr/bin/env python3
"""
Environment setup script for Grok Mimic Training
This script sets up the conda environment and installs required packages.
"""

import subprocess
import sys
import os

def run_command(command, description):
    print(f"Running: {description}")
    print(f"Command: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("Success!")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Output: {e.output}")
        print(f"Error output: {e.stderr}")
        return None

def main():
    print("Setting up environment for Grok Mimic Training...")

    # Check if conda is available
    conda_path = r"X:\Miniconda\Scripts\conda.exe"
    if not os.path.exists(conda_path):
        print("Conda not found at X:\\Miniconda\\Scripts\\conda.exe")
        print("Please install Miniconda or update the path.")
        return

    # Create environment
    env_name = "smallmodel"
    print(f"Creating conda environment: {env_name}")
    run_command(f'"{conda_path}" create -n {env_name} python=3.10 -y', "Create conda environment")

    # Activate environment and install packages
    activate_cmd = f'"{conda_path}" activate {env_name} && '

    packages = [
        "pip install transformers sentencepiece datasets peft accelerate",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
        "pip install bitsandbytes"
    ]

    for package in packages:
        full_cmd = activate_cmd + package
        run_command(full_cmd, f"Install {package}")

    print("Environment setup complete!")
    print(f"Activate the environment with: conda activate {env_name}")
    print("Set HF_TOKEN if needed: set HF_TOKEN=your_token_here")

if __name__ == "__main__":
    main()