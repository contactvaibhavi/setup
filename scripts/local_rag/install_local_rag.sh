#!/bin/bash

# Installation script for Local RAG system
# This makes the script accessible from anywhere in your terminal

set -e

SCRIPT_NAME="local_rag.py"
COMMAND_NAME="rag"
INSTALL_DIR="$HOME/.local/bin"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Installing Local RAG system..."
echo ""

# Create installation directory if it doesn't exist
mkdir -p "$INSTALL_DIR"

# Create wrapper script
cat >"$INSTALL_DIR/$COMMAND_NAME" <<'EOF'
#!/bin/bash
# Wrapper script for local_rag.py

SCRIPT_DIR="$(dirname "$(readlink -f "$0" 2>/dev/null || realpath "$0" 2>/dev/null || echo "$0")")"
ACTUAL_SCRIPT="__INSTALL_PATH__/local_rag.py"

# If the script exists in the expected location, use it
if [ -f "$ACTUAL_SCRIPT" ]; then
    exec python3 "$ACTUAL_SCRIPT" "$@"
else
    echo "❌ Error: local_rag.py not found at $ACTUAL_SCRIPT"
    exit 1
fi
EOF

# Replace placeholder with actual path
sed -i.bak "s|__INSTALL_PATH__|$SCRIPT_DIR|g" "$INSTALL_DIR/$COMMAND_NAME"
rm "$INSTALL_DIR/$COMMAND_NAME.bak" 2>/dev/null || true

# Make executable
chmod +x "$INSTALL_DIR/$COMMAND_NAME"
chmod +x "$SCRIPT_DIR/$SCRIPT_NAME"

echo "✅ Installed wrapper script to: $INSTALL_DIR/$COMMAND_NAME"
echo "✅ Python script location: $SCRIPT_DIR/$SCRIPT_NAME"
echo ""

# Check if directory is in PATH
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
  echo "⚠️  $INSTALL_DIR is not in your PATH"
  echo ""
  echo "Add this line to your ~/.bashrc or ~/.zshrc:"
  echo ""
  echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
  echo ""
  echo "Then run: source ~/.bashrc  (or source ~/.zshrc)"
  echo ""
else
  echo "✅ $INSTALL_DIR is already in your PATH"
  echo ""
fi

echo "🎉 Installation complete!"
echo ""
echo "Usage examples:"
echo "  rag index ~/Documents/PDFs"
echo "  rag ask 'What is machine learning?'"
echo "  rag interactive"
echo "  rag ~/Documents/PDFs 'search query'"
echo ""
