#!/bin/bash
# Textbook PDF Opener
# Opens PDFs from textbooks directory with fuzzy search
TEXTBOOKS_DIR="/Users/vaibhavisingh/Documents/Textbooks"

# Main function
textbook() {
    # No argument - list all
    if [ -z "$1" ]; then
        echo "Available textbooks:"
        find "$TEXTBOOKS_DIR" -type f -name "*.pdf" -exec basename {} \; | nl
        return
    fi
    
    # Search for matching PDF (properly handle spaces)
    local pattern="*${*// /*}*"
    
    # Find matching PDFs and store in array properly
    local results=()
    while IFS= read -r -d '' file; do
        results+=("$file")
    done < <(find "$TEXTBOOKS_DIR" -type f -iname "$pattern.pdf" -print0 2>/dev/null)
    
    case ${#results[@]} in
        0)
            echo "❌ No match for: $*"
            echo ""
            echo "Try one of these:"
            find "$TEXTBOOKS_DIR" -type f -name "*.pdf" -exec basename {} \; | head -5
            ;;
        1)
            echo "📖 Opening: $(basename "${results[0]}")"
            echo "📁 Path: ${results[0]}"
            open -a Preview "${results[0]}"
            ;;
        *)
            echo "Multiple matches found:"
            local i=1
            for book in "${results[@]}"; do
                echo "  [$i] $(basename "$book")"
                ((i++))
            done
            echo ""
            read -p "Select number (1-${#results[@]}): " selection
            
            if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le "${#results[@]}" ]; then
                local selected="${results[$((selection-1))]}"
                echo "📖 Opening: $(basename "$selected")"
                echo "📁 Path: $selected"
                open -a Preview "$selected"
            else
                echo "Invalid selection"
            fi
            ;;
    esac
}

# Run the function with all arguments
textbook "$@"