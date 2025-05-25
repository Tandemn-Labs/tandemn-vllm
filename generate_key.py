from iroh import iroh_ffi
from pathlib import Path

# Generate new Ed25519 private key
key = iroh_ffi.private_key_generate()
key_path = Path("./iroh_peer.key")
key_path.write_bytes(key.encode())
print("🔐 Saved peer key to:", key_path)
