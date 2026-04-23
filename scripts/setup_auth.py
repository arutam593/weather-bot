"""Add or update a user in config/auth.yaml without wiping existing users."""
import getpass
import secrets
from pathlib import Path

import yaml
import streamlit_authenticator as stauth

CONFIG_PATH = Path("config/auth.yaml")

# Load existing config if present, otherwise start fresh
if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f) or {}
    print(f"Loaded existing config with "
          f"{len(config.get('credentials', {}).get('usernames', {}))} user(s).")
else:
    config = {
        "credentials": {"usernames": {}},
        "cookie": {
            "name": "weather_bot_auth",
            "key": secrets.token_hex(32),
            "expiry_days": 30,
        },
        "preauthorized": {"emails": []},
    }
    print("No existing config found — creating new one.")

# Make sure all the required structure is in place
config.setdefault("credentials", {}).setdefault("usernames", {})
config.setdefault("cookie", {
    "name": "weather_bot_auth",
    "key": secrets.token_hex(32),
    "expiry_days": 30,
})

print("\nAdd or update a user.")
username = input("Username: ").strip()
if not username:
    print("Username cannot be empty.")
    raise SystemExit(1)

existing = config["credentials"]["usernames"].get(username)
if existing:
    print(f"User '{username}' already exists. This will UPDATE their info.")
    confirm = input("Continue? (y/N): ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        raise SystemExit(0)

name = input("Display name: ").strip()
email = input("Email: ").strip()

pw1 = getpass.getpass("Password: ")
pw2 = getpass.getpass("Confirm password: ")
if pw1 != pw2:
    print("Passwords do not match. Aborting.")
    raise SystemExit(1)
if len(pw1) < 6:
    print("Password too short (need 6+ chars). Aborting.")
    raise SystemExit(1)

hashed = stauth.Hasher.hash(pw1)

config["credentials"]["usernames"][username] = {
    "name": name,
    "email": email,
    "password": hashed,
}

CONFIG_PATH.parent.mkdir(exist_ok=True)
with open(CONFIG_PATH, "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)

action = "Updated" if existing else "Added"
print(f"\n{action} user '{username}' (display name: {name}).")
print(f"Total users: {len(config['credentials']['usernames'])}")