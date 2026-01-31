# dotfiles: Reproducible Research Environment

**Infrastructure as Code for Systems & NLP Research**

This repository contains the declarative configuration and automation scripts required to provision a consistent, high-velocity research environment across local machines (macOS) and remote compute clusters (Linux/Slurm).

The primary objective is to eliminate environment drift and enable privacy-preserving local inference for sensitive datasets.

## System Architecture

The repository is structured to separate package declaration from installation logic, ensuring modularity and easier auditing of system dependencies.

```text
.
├── manifests/          # Declarative lists of system packages (Brew, Python, Obsidian)
├── llm-config/         # Local LLM model weights and version-controlled configurations
├── scripts/            # Idempotent automation logic
│   ├── install_file.sh # Module-level installer
│   ├── install_folder.sh # Batch provisioning utility
│   └── local_rag/      # Custom pipeline for offline literature retrieval
├── configs/            # Application-specific configurations (Tmux, Alacritty)
└── keyboard/           # Low-level input remapping