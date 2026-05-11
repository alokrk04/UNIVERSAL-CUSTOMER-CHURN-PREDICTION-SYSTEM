#!/usr/bin/env python3
import os
import shutil
import subprocess
import tarfile
import urllib.request
import lzma
import sys

# Clean up
node_dir = os.path.expanduser("~/.local/opt/node")
if os.path.exists(node_dir):
    shutil.rmtree(node_dir)

os.makedirs(node_dir, exist_ok=True)
os.makedirs("/tmp/node_install2", exist_ok=True)

# Download Node.js v20 LTS for macOS arm64
node_version = "v20.11.0"
node_url = f"https://nodejs.org/dist/{node_version}/node-{node_version}-darwin-arm64.tar.xz"
tar_path = "/tmp/node_install2/node.tar.xz"

print(f"Downloading Node.js {node_version}...")
try:
    urllib.request.urlretrieve(node_url, tar_path)
    print("✅ Download complete")
except Exception as e:
    print(f"❌ Download failed: {e}")
    sys.exit(1)

# Extract  
print("📦 Extracting...")
try:
    with lzma.open(tar_path) as f:
        with tarfile.open(fileobj=f) as tar:
            tar.extractall("/tmp/node_install2")
    
    # Copy the full content
    src_root = f"/tmp/node_install2/node-{node_version}-darwin-arm64"
    for item in os.listdir(src_root):
        src = os.path.join(src_root, item)
        dst = os.path.join(node_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    
    print(f"✅ Node.js installed to {node_dir}")
    result = subprocess.check_output(f"{node_dir}/bin/node --version", shell=True, text=True).strip()
    print(f"   Version: {result}")
    
except Exception as e:
    print(f"❌ Extraction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

