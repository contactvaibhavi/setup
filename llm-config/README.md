# Ollama Configuration Backup

**Date:** Sun Jan  4 23:21:31 EST 2026
**Backup Strategy:** Smart backup (configs only, not model files)

## What's Backed Up

- ✅ Model list (`models_20260104.txt`)
- ✅ Custom Modelfiles (if any)
- ✅ RAG vector database (`chroma_db_20260104.tar.gz`)
- ✅ Ollama version info

## What's NOT Backed Up

- ❌ Model files (4-32GB each)
  - Reason: Can be re-downloaded anytime
  - Saves storage space and backup time

## To Restore

### Quick Method (Recommended):
```bash
cd ../ollama_config_backup
./quick_restore.sh
```

### Manual Method:
```bash
# Install each model from list
ollama pull qwen2.5:14b
ollama pull llama3.1
# ... etc

# Restore RAG database
tar -xzf chroma_db_20260104.tar.gz
```

## File Sizes
```
160M	../ollama_config_backup
```

## Current Models
```
NAME          ID              SIZE     MODIFIED          
gemma2:27b    53261bc9c192    15 GB    About an hour ago    
```

---
💡 **Tip:** Sync this folder to iCloud/Dropbox for cloud backup
