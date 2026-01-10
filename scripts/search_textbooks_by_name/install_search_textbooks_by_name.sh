#!/bin/bash

# Installer for Textbook Script
# Adds textbook command to PATH and creates aliases

echo "================================================"
echo "  Textbook Script Installer"
echo "================================================"
echo ""

# Define paths
SCRIPT_NAME="search_textbooks_by_name.sh"
INSTALL_DIR="$HOME/bin"
SCRIPT_PATH="$INSTALL_DIR/textbook"
ZSHRC="$HOME/.zshrc"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Step 1: Create ~/bin directory if it doesn't exist
echo -n "Creating $INSTALL_DIR... "
mkdir -p "$INSTALL_DIR"
echo -e "${GREEN}✓${NC}"

# Step 2: Copy script to ~/bin/textbook
echo -n "Installing textbook script... "
if [ ! -f "$SCRIPT_NAME" ]; then
  echo -e "${YELLOW}✗${NC}"
  echo "Error: $SCRIPT_NAME not found in current directory"
  exit 1
fi

cp "$SCRIPT_NAME" "$SCRIPT_PATH"
chmod +x "$SCRIPT_PATH"
echo -e "${GREEN}✓${NC}"

# Step 3: Add ~/bin to PATH if not already there
echo -n "Checking PATH... "
if ! grep -q 'export PATH="$HOME/bin:$PATH"' "$ZSHRC" 2>/dev/null; then
  echo ""
  echo "Adding ~/bin to PATH in ~/.zshrc..."
  cat >>"$ZSHRC" <<'PATHEOF'

# Add ~/bin to PATH
export PATH="$HOME/bin:$PATH"
PATHEOF
  echo -e "${GREEN}✓${NC} Added to PATH"
else
  echo -e "${GREEN}✓${NC} Already in PATH"
fi

# Step 4: Add aliases to ~/.zshrc
echo -n "Adding aliases... "
if ! grep -q 'alias tb=' "$ZSHRC" 2>/dev/null; then
  cat >>"$ZSHRC" <<'ALIASEOF'

# Textbook shortcuts
export TEXTBOOKS_DIR="/Users/vaibhavisingh/Documents/Textbooks"
alias tb='textbook'
alias tbl='find "$TEXTBOOKS_DIR" -type f -name "*.pdf" -exec basename {} \;'
ALIASEOF
  echo -e "${GREEN}✓${NC}"
else
  echo -e "${YELLOW}⚠${NC}  Aliases already exist"
fi

# Step 5: Summary
echo ""
echo "================================================"
echo -e "${BLUE}Installation Complete!${NC}"
echo "================================================"
echo ""
echo "Installed to: $SCRIPT_PATH"
echo ""
echo "Available commands:"
echo "  textbook [search]  - Search and open textbook"
echo "  tb [search]        - Short alias"
echo "  tbl                - List all textbooks"
echo ""
echo -e "${YELLOW}Important:${NC} Run this command to use immediately:"
echo "  source ~/.zshrc"
echo ""
echo "Or restart your terminal."
echo ""
