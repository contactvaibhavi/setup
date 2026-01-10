# Script Manager

Easily install scripts to `~/bin` and make them executable.

## Structure
```
scripts/
├── install_script.sh       # Install single script
├── install_all.sh          # Install all scripts in scripts/
├── specific_scripts/                # Put your scripts here
│   └── specific_script.sh
└── README.md
```

## Usage

### Install Single Script
```bash
# Install with original name (minus .sh)
./install_script.sh specific_scripts/specific_script.sh

# Install with custom name
./install_script.sh specific_scripts/specific_script.sh custom-name
```

### Install All Scripts
```bash
./install_all.sh
```

## Examples

### Install Obsidian Fullscreen Script
```bash
./install-script.sh scripts/open-obsidian-fullscreen.sh obsidian-fullscreen
```

Then use:
```bash
obsidian-fullscreen
```

### Install Textbook Search By Name
```bash
./install-script.sh ../Obsidian/search_textbooks_by_name/search_textbooks_by_name.sh textbook
```

## Notes

- Scripts are copied to `~/bin`
- Made executable automatically
- Original files remain in the `scripts/` folder
- Ensure `~/bin` is in your PATH
