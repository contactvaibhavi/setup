#!/bin/bash
# Uninstallation script for search_pdfs

INSTALL_PATH="/usr/local/bin/search_pdfs"

echo "Uninstalling search_pdfs..."

# Check if symlink exists
if [ -L "$INSTALL_PATH" ]; then
  sudo rm "$INSTALL_PATH"
  echo "✅ Uninstallation successful!"
  echo "search_pdfs has been removed from /usr/local/bin"
else
  echo "⚠️  search_pdfs is not installed"
fi
