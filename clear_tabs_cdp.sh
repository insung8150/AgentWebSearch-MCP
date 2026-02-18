#!/bin/bash
# Chrome Tab Cleanup via CDP

echo "=== Chrome Tab Cleanup (CDP Direct) ==="

# Current tab count
count=$(curl -s http://localhost:9222/json/list | jq length)
echo "Current tabs: ${count}"

if [ "$count" -le 1 ]; then
    echo "No cleanup needed"
    exit 0
fi

# Create blank tab (prevent Chrome from closing)
echo "Creating blank tab..."
curl -s "http://localhost:9222/json/new?about:blank" > /dev/null

# Get all tab IDs (except last)
echo "Closing tabs..."
ids=$(curl -s http://localhost:9222/json/list | jq -r '.[:-1][].id')

closed=0
for id in $ids; do
    curl -s "http://localhost:9222/json/close/$id" > /dev/null
    ((closed++))
    if [ $((closed % 10)) -eq 0 ]; then
        echo "  ${closed} closed..."
    fi
done

# Final check
final=$(curl -s http://localhost:9222/json/list | jq length)
echo ""
echo "Cleanup complete: ${final} tabs remaining"
