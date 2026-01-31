#!/bin/bash

# Script Installer
# Installs any script to ~/bin and makes it executable

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

INSTALL_DIR="$HOME/bin"

echo "================================================"
echo "  Script Installer"
echo "================================================"
echo ""

# Check if script file was provided
if [ -z "$1" ]; then
    echo -e "${RED}Error:${NC} No script provided"
    echo ""
    echo "Usage: $0 <script_file> [install_name]"
    echo ""
    echo "Examples:"
    echo "  $0 my-script.sh"
    echo "  $0 my-script.sh custom-name"
    echo "  $0 scripts/open-obsidian-fullscreen.sh"
    echo ""
    exit 1
fi

SCRIPT_FILE="$1"
SCRIPT_NAME="${2:-$(basename "$SCRIPT_FILE" .sh)}"  # Use provided name or strip .sh

# Check if source script exists
if [ ! -f "$SCRIPT_FILE" ]; then
    echo -e "${RED}✗${NC} Script not found: $SCRIPT_FILE"
    exit 1
fi

# Create ~/bin if it doesn't exist
mkdir -p "$INSTALL_DIR"

# Copy script to ~/bin
INSTALL_PATH="$INSTALL_DIR/$SCRIPT_NAME"

echo -n "Installing $SCRIPT_NAME... "
cp "$SCRIPT_FILE" "$INSTALL_PATH"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

# Make executable
echo -n "Making executable... "
chmod +x "$INSTALL_PATH"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

# Check if ~/bin is in PATH
echo -n "Checking PATH... "
if [[ ":$PATH:" == *":$HOME/bin:"* ]]; then
    echo -e "${GREEN}✓${NC} Already in PATH"
else
    echo -e "${YELLOW}⚠${NC}  Not in PATH"
    echo ""
    echo "Add this to your ~/.zshrc:"
    echo '  export PATH="$HOME/bin:$PATH"'
    echo ""
    echo "Then run: source ~/.zshrc"
fi

# Summary
echo ""
echo "================================================"
echo -e "${BLUE}Installation Complete!${NC}"
echo "================================================"
echo ""
echo "Script: $SCRIPT_NAME"
echo "Location: $INSTALL_PATH"
echo ""
echo "To use:"
echo "  $SCRIPT_NAME [arguments]"
echo ""

