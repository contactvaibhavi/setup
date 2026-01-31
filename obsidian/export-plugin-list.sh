#!/bin/bash

VAULT="/Users/vaibhavisingh/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian"
OUTPUT="$HOME/obsidian-plugins-$(date +%Y%m%d).txt"

echo "Exporting plugin list..."

{
    echo "Obsidian Plugins - $(date)"
    echo "================================"
    echo ""
    
    for plugin in "$VAULT/.obsidian/plugins"/*; do
        if [ -d "$plugin" ]; then
            manifest="$plugin/manifest.json"
            if [ -f "$manifest" ]; then
                name=$(grep '"name"' "$manifest" | head -1 | sed 's/.*"name": "\(.*\)".*/\1/')
                version=$(grep '"version"' "$manifest" | head -1 | sed 's/.*"version": "\(.*\)".*/\1/')
                echo "- $name (v$version)"
            fi
        fi
    done
    
    echo ""
    echo "Total: $(ls -1 "$VAULT/.obsidian/plugins" | wc -l | tr -d ' ') plugins"
} > "$OUTPUT"

echo "✓ Plugin list saved to: $OUTPUT"
cat "$OUTPUT"

