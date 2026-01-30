#!/bin/bash
# Textbook PDF Opener
# Opens PDFs from textbooks directory with fuzzy search
TEXTBOOKS_DIR="/Users/vaibhavisingh/Documents/Textbooks"

# Main function
textbook() {
    # No argument - list all
    if [ -z "$1" ]; then
        echo "Available textbooks:"
        find "$TEXTBOOKS_DIR" -type f -name "*.pdf" -print0 | while IFS= read -r -d '' file; do
            local rel_path="${file#$TEXTBOOKS_DIR/}"
            printf "%s\n  📁 %s\n" "$(basename "$file")" "$rel_path"
        done | nl -s '. ' -w 3
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
            local matched_file="${results[0]}"
            local rel_path="${matched_file#$TEXTBOOKS_DIR/}"
            printf "📖 Opening: %s\n" "$(basename "$matched_file")"
            printf "📁 Path: %s\n\n" "$rel_path"
            sleep 0.1
            open -a Preview "$matched_file"
            ;;
        *)
            echo "Multiple matches found:"
            local i=1
            for book in "${results[@]}"; do
                local rel_path="${book#$TEXTBOOKS_DIR/}"
                echo "  [$i] $(basename "$book")"
                echo "      📁 $rel_path"
                ((i++))
            done
            echo ""
            read -p "Select number (1-${#results[@]}): " selection
            
            if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le "${#results[@]}" ]; then
                local selected="${results[$((selection-1))]}"
                local rel_path="${selected#$TEXTBOOKS_DIR/}"
                printf "\n📖 Opening: %s\n" "$(basename "$selected")"
                printf "📁 Path: %s\n\n" "$rel_path"
                sleep 0.1
                open -a Preview "$selected"
            else
                echo "Invalid selection"
            fi
            ;;
    esac
}

# Run the function with all arguments
textbook "$@"