"""Remove a user from config/auth.yaml."""
from pathlib import Path

import yaml

CONFIG_PATH = Path("config/auth.yaml")

if not CONFIG_PATH.exists():
    print(f"No config file at {CONFIG_PATH}. Nothing to remove.")
    raise SystemExit(1)

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f) or {}

users = config.get("credentials", {}).get("usernames", {})
if not users:
    print("No users in config. Nothing to remove.")
    raise SystemExit(0)

print("\nCurrent users:")
for u, info in users.items():
    print(f"  - {u}  ({info.get('name', '?')}, {info.get('email', '?')})")

print()
username = input("Username to remove: ").strip()
if not username:
    print("No username given. Aborting.")
    raise SystemExit(1)

if username not in users:
    print(f"User '{username}' not found. Aborting.")
    raise SystemExit(1)

if len(users) == 1:
    print(f"\n⚠ '{username}' is the ONLY user. Removing them will lock everyone out.")
    print("  You'd have to re-run setup_auth.py to create a new account.")
    confirm = input("Really remove? (type YES to confirm): ").strip()
    if confirm != "YES":
        print("Aborted.")
        raise SystemExit(0)
else:
    confirm = input(f"Remove '{username}' ({users[username].get('name', '?')})? (y/N): ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        raise SystemExit(0)

del config["credentials"]["usernames"][username]

with open(CONFIG_PATH, "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)

print(f"\nRemoved user '{username}'.")
print(f"Remaining users: {len(config['credentials']['usernames'])}")