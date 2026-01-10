#!/bin/bash
# Installation script for search_pdfs

SCRIPT_PATH="$HOME/gitCode/setup/scripts/search_pdfs.py"
INSTALL_PATH="/usr/local/bin/search_pdfs"

echo "Installing search_pdfs..."

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
  echo "Error: Script not found at $SCRIPT_PATH"
  exit 1
fi

# Make script executable
echo "Making script executable..."
chmod +x "$SCRIPT_PATH"

# Create symlink
echo "Creating symlink in /usr/local/bin..."
sudo ln -sf "$SCRIPT_PATH" "$INSTALL_PATH"

# Verify installation
if [ -L "$INSTALL_PATH" ]; then
  echo "✅ Installation successful!"
  echo "You can now run: search_pdfs 'your search term' /path/to/pdfs"
  echo ""
  echo "Usage examples:"
  echo "  search_pdfs 'majority element' ~/Documents/Textbooks/Algorithms"
  echo "  search_pdfs 'dynamic programming' ~/Documents --all"
  echo "  search_pdfs 'graph algorithm' ~/Documents --threads 8 --html"
else
  echo "❌ Installation failed"
  exit 1
fi
