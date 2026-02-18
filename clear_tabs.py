#!/usr/bin/env python3
"""
Chrome Tab Cleanup Utility

Closes all tabs via CDP API and leaves only one blank tab.
Chrome session (cookies etc.) is preserved.
"""

import httpx
import time


def clear_all_tabs(chrome_port: int = 9222) -> bool:
    """
    Close all tabs (using CDP API directly)

    Args:
        chrome_port: Chrome DevTools port (default 9222)

    Returns:
        Success status
    """
    base_url = f"http://localhost:{chrome_port}"
    print(f"=== Chrome Tab Cleanup (port {chrome_port}) ===\n")

    try:
        # Get current tab list
        resp = httpx.get(f"{base_url}/json/list", timeout=5)
        tabs = resp.json()
        count = len(tabs)
        print(f"Current tabs: {count}")

        if count <= 1:
            print("No cleanup needed (1 or fewer tabs)")
            return True

        # Create blank tab (to prevent Chrome from closing when all tabs are closed)
        print("Creating blank tab...")
        httpx.get(f"{base_url}/json/new?about:blank", timeout=5)
        time.sleep(0.3)

        # Close all existing tabs
        print(f"Closing {count} tabs...")
        closed = 0
        for tab in tabs:
            tab_id = tab.get("id")
            if tab_id:
                try:
                    httpx.get(f"{base_url}/json/close/{tab_id}", timeout=5)
                    closed += 1
                    time.sleep(0.1)
                    if closed % 10 == 0:
                        print(f"  {closed} closed...")
                except Exception as e:
                    print(f"  Failed to close tab (ignored): {e}")

        # Final check
        resp = httpx.get(f"{base_url}/json/list", timeout=5)
        final = len(resp.json())
        print(f"\nCleanup complete: {final} tabs remaining")
        return True

    except httpx.ConnectError:
        print(f"Failed to connect to Chrome CDP (port {chrome_port})")
        print("   Ensure Chrome is running with --remote-debugging-port=9222")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9222
    clear_all_tabs(port)
