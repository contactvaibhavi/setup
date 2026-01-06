# Local RAG - Work From Anywhere 🚀

A fully local RAG (Retrieval Augmented Generation) system that works from any directory in your terminal.

## Features
- ✅ Run from anywhere - no need to be in a specific directory
- ✅ Fully local - no API keys, no internet needed
- ✅ Fast semantic search with hybrid ranking
- ✅ Conversation history and context reuse
- ✅ Interactive and one-shot modes
- ✅ Auto-opens PDFs in Preview

## Quick Installation

### 1. Install dependencies

```bash
pip install sentence-transformers chromadb PyPDF2 requests
```

### 2. Install Ollama

```bash
# macOS
brew install ollama

# Start Ollama
ollama serve

# In another terminal, pull a model
ollama pull gemma2:27b
```

### 3. Download the script

Save `local_rag.py` to a directory (e.g., `~/scripts/`)

### 4. Run the installer

```bash
cd ~/scripts  # or wherever you saved the files
chmod +x install.sh
./install.sh
```

### 5. Add to PATH (if needed)

If the installer says `~/.local/bin` is not in your PATH, add this to your `~/.bashrc` or `~/.zshrc`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Then reload your shell:

```bash
source ~/.bashrc  # or source ~/.zshrc
```

## Usage

### Quick Search (Recommended)
Search and ask questions in one command:

```bash
rag ~/Documents/PDFs "What is dynamic programming?"
```

This will:
1. Index the PDFs if not already indexed
2. Search for relevant content
3. Generate an answer with context
4. Offer to open source PDFs

### Index PDFs
Create a searchable index:

```bash
rag index ~/Documents/Textbooks
```

### Ask Questions
Query your indexed PDFs:

```bash
rag ask "Explain the Boyer-Moore algorithm"
```

### Interactive Mode
Start a conversation:

```bash
rag interactive
```

Commands in interactive mode:
- Type questions naturally
- `list` - Show all indexed PDFs
- `debug: <query>` - Debug search results
- `setfolder <path>` - Set PDF folder location
- `quit` - Exit

### Debug Search
See what's being found:

```bash
rag debug "machine learning"
```

### List Indexed PDFs

```bash
rag list
```

## How It Works

### Data Storage
All data is stored in `~/.local/share/local_rag/`:
- `chroma_db/` - Vector database
- `chroma_db/.pdf_folder` - Remembered PDF folder location

This means you can run `rag` from anywhere and it will remember your PDFs.

### Search Process
1. **Text Extraction**: Reads PDFs and normalizes text
2. **Chunking**: Splits into 500-char chunks with 50-char overlap
3. **Embedding**: Uses `all-MiniLM-L6-v2` for semantic vectors
4. **Hybrid Search**: Combines semantic similarity + keyword matching
5. **LLM Generation**: Ollama generates answers from context

## Advanced Usage

### Conversation Context
The system maintains context across questions:

```bash
rag ask "What is recursion?"
# Follow-up keeps context:
q  # Type 'q' to ask another question with same context
n  # Type 'n' for fresh search
```

### Open Specific PDFs
After getting an answer, you can:
- Enter `1-3` to open that source PDF
- Enter `q` to ask follow-up (keeps context)
- Enter `n` for new search (fresh context)
- Press Enter to finish

### Different Models
Use different Ollama models:

```python
# Edit local_rag.py and change:
model="phi4"  # Faster, less capable
model="gemma2:27b"  # Default, good balance
model="llama3.1"  # Larger, more capable
```

## Examples

### Quick search in a folder
```bash
rag ~/Documents/CS_Books "explain quicksort"
```

### Index multiple PDF folders
```bash
rag index ~/Documents/Math_Books
rag index ~/Documents/CS_Books
rag list  # See all indexed PDFs
```

### Debug why a search isn't working
```bash
rag debug "neural networks"
```

### Interactive session
```bash
rag interactive

💬 Question: What is backpropagation?
# Answer appears with sources

💬 Question: list
# Shows all indexed PDFs

💬 Question: How does gradient descent work?
# Continues with context

💬 Question: quit
```

## Troubleshooting

### Command not found: rag
```bash
# Check if ~/.local/bin is in PATH
echo $PATH | grep ".local/bin"

# If not, add to ~/.bashrc or ~/.zshrc:
export PATH="$HOME/.local/bin:$PATH"
source ~/.bashrc
```

### Cannot connect to Ollama
```bash
# Start Ollama
ollama serve

# In another terminal:
ollama list  # Check installed models
ollama pull gemma2:27b  # Install if needed
```

### PDF folder not found
```bash
# Re-index to set folder location
rag index ~/Documents/PDFs

# Or set manually
rag setfolder ~/Documents/PDFs
```

### Slow indexing
The script processes PDFs in parallel. For faster indexing:
- Reduce `chunk_size` in the code (default: 500)
- Increase `num_workers` (default: 4)
- Use SSD storage

### No results found
```bash
# Debug the search
rag debug "your search query"

# Check if PDFs are indexed
rag list

# Try reindexing
rag index ~/your/pdf/folder
```

## Uninstall

```bash
# Remove the command
rm ~/.local/bin/rag

# Remove data (optional)
rm -rf ~/.local/share/local_rag
```

## Tips

1. **Use specific queries**: "explain quicksort algorithm" works better than "quicksort"
2. **Context matters**: Use 'q' to keep context for follow-up questions
3. **Debug searches**: Use `rag debug "<query>"` to see what's being found
4. **Multiple folders**: Index different PDF folders separately
5. **Interactive mode**: Best for exploring a topic with multiple questions

## Technical Details

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Vector DB**: ChromaDB with persistence
- **LLM**: Local Ollama (configurable model)
- **Chunk Size**: 500 characters with 50-char overlap
- **Search**: Hybrid (semantic + keyword matching)
- **Speed**: ~10-50 chunks/second depending on hardware

## License

MIT - Free to use and modify!
