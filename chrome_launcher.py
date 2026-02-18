#!/usr/bin/env python3
"""
Chrome Multi-Instance Launcher

Manages independent Chrome instances per portal
- Separate CDP port for each portal
- Session (cookies/login) persistence via user-data-dir
- Start/stop/status check
"""

import subprocess
import time
import httpx
from pathlib import Path

# ========== Chrome Instance Configuration per Portal ==========
#
# [Portal Addition Guide - Claude Code can read this and add automatically]
#
# To add a new portal:
# 1. Add new entry to CHROME_INSTANCES below
# 2. Add matching key to PORTAL_CONFIG in cdp_search.py
#
# Required information:
# - port: CDP port (avoid conflicts with existing ports, sequential from 9222)
# - profile: Chrome profile path (for session/cookie persistence)
# - start_url: URL opened when Chrome starts
#
# Currently used ports:
# - 9222: naver
# - 9223: google
# - 9224: brave
# - 9225+: for new portals (sequential allocation)
#
# Example - Adding Bing:
# "bing": {
#     "port": 9225,
#     "profile": "/tmp/chrome-bing-profile",
#     "start_url": "https://www.bing.com"
# },
#
# Example - Adding DuckDuckGo:
# "duckduckgo": {
#     "port": 9226,
#     "profile": "/tmp/chrome-duckduckgo-profile",
#     "start_url": "https://duckduckgo.com"
# },
#
# Example - Adding Baidu (China):
# "baidu": {
#     "port": 9227,
#     "profile": "/tmp/chrome-baidu-profile",
#     "start_url": "https://www.baidu.com"
# },
#
# Example - Adding Yandex (Russia):
# "yandex": {
#     "port": 9228,
#     "profile": "/tmp/chrome-yandex-profile",
#     "start_url": "https://yandex.ru"
# },
#
# Notes:
# - Port numbers must not conflict
# - Profile path should be different for each portal (session isolation)
# - More Chrome instances = more memory usage
# - Use portals parameter to select only needed portals
#

CHROME_INSTANCES = {
    "naver": {
        "port": 9222,
        "profile": "/tmp/chrome-naver-profile",
        "start_url": "https://www.naver.com/"  # Integrated search
    },
    "google": {
        "port": 9223,
        "profile": "/tmp/chrome-google-profile",
        "start_url": "https://www.google.com"
    },
    "brave": {
        "port": 9224,
        "profile": "/tmp/chrome-brave-profile",
        "start_url": "https://search.brave.com"
    }
}


def is_chrome_running(port: int) -> bool:
    """Check if Chrome is running on the specified port"""
    try:
        resp = httpx.get(f"http://localhost:{port}/json/version", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def start_chrome(portal: str) -> bool:
    """
    Start a single Chrome instance

    Args:
        portal: "naver", "google", "brave"

    Returns:
        Success status
    """
    config = CHROME_INSTANCES.get(portal)
    if not config:
        print(f"[{portal}] Unknown portal")
        return False

    port = config["port"]
    profile = config["profile"]
    start_url = config["start_url"]

    # Skip if already running
    if is_chrome_running(port):
        print(f"[{portal}] Already running (port {port})")
        return True

    # Create profile directory
    Path(profile).mkdir(parents=True, exist_ok=True)

    # Start Chrome (with detection evasion flags)
    cmd = [
        "google-chrome",
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile}",
        "--remote-allow-origins=*",  # Allow WebSocket connections
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
        # Detection evasion flags
        "--disable-blink-features=AutomationControlled",
        "--disable-dev-shm-usage",
        "--disable-infobars",
        "--disable-extensions",
        "--disable-popup-blocking",
        "--ignore-certificate-errors",
        start_url
    ]

    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )

        # Wait for startup
        for i in range(10):
            time.sleep(0.5)
            if is_chrome_running(port):
                print(f"[{portal}] Started (port {port})")
                return True

        print(f"[{portal}] Start failed (timeout)")
        return False

    except Exception as e:
        print(f"[{portal}] Start error: {e}")
        return False


def stop_chrome(portal: str) -> bool:
    """
    Stop a single Chrome instance

    CDP /json/close cannot close the browser itself,
    so we terminate the process directly
    """
    config = CHROME_INSTANCES.get(portal)
    if not config:
        return False

    port = config["port"]

    if not is_chrome_running(port):
        print(f"[{portal}] Not running")
        return True

    try:
        # Find and kill process using the port
        result = subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            capture_output=True,
            timeout=5
        )
        time.sleep(0.5)

        if not is_chrome_running(port):
            print(f"[{portal}] Stopped")
            return True
        else:
            print(f"[{portal}] Stop failed")
            return False

    except Exception as e:
        print(f"[{portal}] Stop error: {e}")
        return False


def start_all() -> dict:
    """Start all Chrome instances"""
    print("=== Starting 3 Chrome instances ===\n")

    results = {}
    for portal in CHROME_INSTANCES:
        results[portal] = start_chrome(portal)
        time.sleep(0.5)  # Sequential start

    print()
    return results


def stop_all() -> dict:
    """Stop all Chrome instances"""
    print("=== Stopping 3 Chrome instances ===\n")

    results = {}
    for portal in CHROME_INSTANCES:
        results[portal] = stop_chrome(portal)

    print()
    return results


def status() -> dict:
    """Check status of all Chrome instances"""
    print("=== Chrome Status ===\n")

    results = {}
    for portal, config in CHROME_INSTANCES.items():
        port = config["port"]
        running = is_chrome_running(port)
        results[portal] = running

        status_str = "Running" if running else "Stopped"
        print(f"[{portal}] port {port}: {status_str}")

    print()
    return results


def get_port(portal: str) -> int:
    """Return CDP port for portal"""
    config = CHROME_INSTANCES.get(portal)
    return config["port"] if config else 0


# ========== CLI ==========

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python chrome_launcher.py [start|stop|status|restart]")
        print()
        status()
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "start":
        start_all()
        status()
    elif cmd == "stop":
        stop_all()
    elif cmd == "status":
        status()
    elif cmd == "restart":
        stop_all()
        time.sleep(1)
        start_all()
        status()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
