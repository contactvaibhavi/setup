#!/bin/bash
# Quick restore - pulls models from latest list

echo "🔄 Restoring Ollama Setup"
echo "========================"
echo ""

# Find latest model list
LATEST_MODELS=$(ls -t models_*.txt 2>/dev/null | head -1)

if [ -z "$LATEST_MODELS" ]; then
    echo "❌ No model list found"
    exit 1
fi

echo "📋 Using model list: $LATEST_MODELS"
echo ""
echo "Models to install:"
cat "$LATEST_MODELS"
echo ""

read -p "Install these models? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Parse and pull each model
    while IFS= read -r line; do
        # Skip header
        if [[ $line =~ ^NAME ]]; then
            continue
        fi
        
        # Extract model name (first column)
        model=$(echo "$line" | awk '{print $1}')
        
        if [[ ! -z "$model" ]]; then
            echo ""
            echo "⬇️  Pulling $model..."
            ollama pull "$model"
        fi
    done < "$LATEST_MODELS"
    
    echo ""
    echo "✅ Models restored!"
fi

# Restore RAG database if exists
LATEST_DB=$(ls -t chroma_db_*.tar.gz 2>/dev/null | head -1)
if [ ! -z "$LATEST_DB" ]; then
    echo ""
    read -p "Restore RAG database? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tar -xzf "$LATEST_DB"
        echo "✅ RAG database restored"
    fi
fi

echo ""
echo "✅ Restore complete!"
