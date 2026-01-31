#!/bin/bash

# Install All Scripts in scripts/ folder

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)/scripts"
INSTALLER="$(cd "$(dirname "$0")" && pwd)/install-script.sh"

echo "================================================"
echo "  Batch Script Installer"
echo "================================================"
echo ""

if [ ! -d "$SCRIPT_DIR" ]; then
    echo "Error: scripts/ directory not found"
    exit 1
fi

# Count scripts
script_count=$(find "$SCRIPT_DIR" -type f -name "*.sh" | wc -l | tr -d ' ')

if [ "$script_count" -eq 0 ]; then
    echo "No scripts found in $SCRIPT_DIR"
    exit 0
fi

echo "Found $script_count script(s) to install"
echo ""

# Install each script
for script in "$SCRIPT_DIR"/*.sh; do
    if [ -f "$script" ]; then
        echo "Installing: $(basename "$script")"
        "$INSTALLER" "$script"
        echo ""
    fi
done

echo "================================================"
echo -e "${BLUE}All Scripts Installed!${NC}"
echo "================================================"
echo ""

