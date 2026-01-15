#!/bin/bash

# LaTeX Compiler Script
# Usage: ./compile_latex.sh [main.tex] [options]
# Options:
#   -o, --open    Open PDF after compilation
#   -c, --clean   Clean auxiliary files after compilation
#   -w, --watch   Watch mode - recompile on file changes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
MAIN_FILE="main.tex"
OPEN_PDF=false
CLEAN_AUX=false
WATCH_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--open)
            OPEN_PDF=true
            shift
            ;;
        -c|--clean)
            CLEAN_AUX=true
            shift
            ;;
        -w|--watch)
            WATCH_MODE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [main.tex] [options]"
            echo "Options:"
            echo "  -o, --open    Open PDF after compilation"
            echo "  -c, --clean   Clean auxiliary files after compilation"
            echo "  -w, --watch   Watch mode - recompile on file changes"
            echo "  -h, --help    Show this help message"
            exit 0
            ;;
        *.tex)
            MAIN_FILE="$1"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if file exists
if [ ! -f "$MAIN_FILE" ]; then
    echo -e "${RED}Error: File '$MAIN_FILE' not found${NC}"
    exit 1
fi

# Get basename without extension
BASENAME="${MAIN_FILE%.tex}"

# Function to compile LaTeX
compile_latex() {
    echo -e "${YELLOW}Compiling $MAIN_FILE...${NC}"
    
    # Run pdflatex twice for references and TOC
    # -interaction=nonstopmode continues on errors without prompting
    # -file-line-error shows error locations
    pdflatex -interaction=nonstopmode -file-line-error -halt-on-error "$MAIN_FILE"
    
    if [ $? -eq 0 ]; then
        # Run again for cross-references
        pdflatex -interaction=nonstopmode -file-line-error -halt-on-error "$MAIN_FILE" > /dev/null
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Compilation successful: ${BASENAME}.pdf${NC}"
            
            # Clean auxiliary files if requested
            if [ "$CLEAN_AUX" = true ]; then
                clean_files
            fi
            
            # Open PDF if requested
            if [ "$OPEN_PDF" = true ]; then
                open_pdf
            fi
            
            return 0
        else
            echo -e "${RED}✗ Compilation failed (second pass)${NC}"
            echo -e "${YELLOW}Check the output above for errors${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ Compilation failed${NC}"
        echo -e "${YELLOW}Check the output above for errors${NC}"
        return 1
    fi
}

# Function to clean auxiliary files
clean_files() {
    echo -e "${YELLOW}Cleaning auxiliary files...${NC}"
    rm -f "${BASENAME}.aux" "${BASENAME}.log" "${BASENAME}.out" \
          "${BASENAME}.toc" "${BASENAME}.lof" "${BASENAME}.lot" \
          "${BASENAME}.bbl" "${BASENAME}.blg" "${BASENAME}.fls" \
          "${BASENAME}.fdb_latexmk" "${BASENAME}.synctex.gz"
    echo -e "${GREEN}✓ Cleaned${NC}"
}

# Function to open PDF
open_pdf() {
    PDF_FILE="${BASENAME}.pdf"
    
    if [ ! -f "$PDF_FILE" ]; then
        echo -e "${RED}Error: PDF file not found${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}Opening PDF...${NC}"
    
    # Detect OS and open accordingly
    case "$(uname -s)" in
        Darwin*)
            # macOS - open in Preview
            open "$PDF_FILE"
            ;;
        Linux*)
            # Linux - try common PDF viewers
            if command -v xdg-open &> /dev/null; then
                xdg-open "$PDF_FILE" &> /dev/null &
            elif command -v evince &> /dev/null; then
                evince "$PDF_FILE" &> /dev/null &
            elif command -v okular &> /dev/null; then
                okular "$PDF_FILE" &> /dev/null &
            else
                echo -e "${RED}No PDF viewer found${NC}"
                return 1
            fi
            ;;
        MINGW*|MSYS*|CYGWIN*)
            # Windows
            start "$PDF_FILE"
            ;;
        *)
            echo -e "${RED}Unsupported operating system${NC}"
            return 1
            ;;
    esac
    
    echo -e "${GREEN}✓ PDF opened${NC}"
}

# Watch mode
if [ "$WATCH_MODE" = true ]; then
    echo -e "${YELLOW}Watch mode enabled. Press Ctrl+C to stop.${NC}"
    echo -e "${YELLOW}Watching: $MAIN_FILE${NC}"
    echo ""
    
    # Initial compilation
    compile_latex
    
    # Check if fswatch is available
    if command -v fswatch &> /dev/null; then
        # Use fswatch (works on macOS and Linux)
        fswatch -o "$MAIN_FILE" | while read f; do
            echo ""
            echo -e "${YELLOW}File changed, recompiling...${NC}"
            compile_latex
        done
    elif command -v inotifywait &> /dev/null; then
        # Use inotifywait (Linux)
        while inotifywait -e modify "$MAIN_FILE" 2>/dev/null; do
            echo ""
            echo -e "${YELLOW}File changed, recompiling...${NC}"
            compile_latex
        done
    else
        echo -e "${RED}Error: Watch mode requires 'fswatch' (macOS/Linux) or 'inotifywait' (Linux)${NC}"
        echo "Install with: brew install fswatch (macOS) or apt-get install inotify-tools (Linux)"
        exit 1
    fi
else
    # Single compilation
    compile_latex
fi