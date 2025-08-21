#!/bin/bash

# Horse ID Pipeline Runner (macOS/Linux)
# 
# This script automatically detects and activates the Python virtual environment,
# then runs the unified horse identification pipeline.
#
# Usage:
#   ./run_pipeline.sh                                    # Directory ingestion (interactive)  
#   ./run_pipeline.sh --email                            # Email ingestion
#   ./run_pipeline.sh --dir --path /path/to/horses --date 20240315  # Non-interactive directory
#   ./run_pipeline.sh --force                            # Force override locks

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to find and activate virtual environment
activate_venv() {
    local venv_paths=(
        "$SCRIPT_DIR/venv"
        "$SCRIPT_DIR/.venv" 
        "$SCRIPT_DIR/../venv"
        "$SCRIPT_DIR/../.venv"
        "./venv"
        "./.venv"
    )
    
    echo "🔍 Searching for Python virtual environment..."
    
    for venv_path in "${venv_paths[@]}"; do
        if [ -f "$venv_path/bin/activate" ]; then
            echo "✅ Found virtual environment: $venv_path"
            source "$venv_path/bin/activate"
            echo "🐍 Virtual environment activated: $(python --version)"
            return 0
        fi
    done
    
    echo "⚠️  No virtual environment found. Trying system Python..."
    echo "🐍 Using system Python: $(python3 --version 2>/dev/null || python --version)"
    
    # Check if required modules are available
    if ! python3 -c "import pandas, yaml" 2>/dev/null && ! python -c "import pandas, yaml" 2>/dev/null; then
        echo "❌ Required Python packages not found."
        echo ""
        echo "Please either:"
        echo "1. Create and activate a virtual environment with required packages"
        echo "2. Install required packages globally: pip install pandas pyyaml pillow streamlit ultralytics"
        echo ""
        echo "Recommended setup:"
        echo "  python -m venv venv"
        echo "  source venv/bin/activate" 
        echo "  pip install -r requirements.txt  # if you have one"
        exit 1
    fi
    
    return 0
}

# Function to determine Python command
get_python_cmd() {
    if command -v python3 >/dev/null 2>&1; then
        echo "python3"
    elif command -v python >/dev/null 2>&1; then
        echo "python"
    else
        echo "❌ Python not found. Please install Python 3.7 or later."
        exit 1
    fi
}

# Main execution
main() {
    echo "🐴 Horse ID Pipeline Runner (macOS/Linux)"
    echo "========================================="
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Activate virtual environment if available
    activate_venv
    
    # Determine Python command
    PYTHON_CMD=$(get_python_cmd)
    
    # Check if run_pipeline.py exists
    if [ ! -f "run_pipeline.py" ]; then
        echo "❌ run_pipeline.py not found in $SCRIPT_DIR"
        echo "Please ensure you're running this script from the horse-id directory."
        exit 1
    fi
    
    echo ""
    echo "🚀 Starting pipeline with arguments: $*"
    echo ""
    
    # Execute the Python pipeline script with all provided arguments
    exec "$PYTHON_CMD" run_pipeline.py "$@"
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n\n🛑 Pipeline interrupted. Cleaning up..."; exit 130' INT

# Show help if requested
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "🐴 Horse ID Pipeline Runner (macOS/Linux)"
    echo ""
    echo "Usage:"
    echo "  $0 [OPTIONS]"
    echo ""
    echo "Ingestion modes:"
    echo "  (default)                           Directory ingestion (interactive)"
    echo "  --email                             Email ingestion"
    echo "  --dir --path PATH --date YYYYMMDD   Non-interactive directory ingestion"
    echo ""
    echo "Options:"
    echo "  --force                             Force override existing locks"
    echo "  --check-lock                        Check lock status and exit"
    echo "  -h, --help                          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                  # Interactive directory ingestion"
    echo "  $0 --email                          # Email ingestion"
    echo "  $0 --dir --path /photos --date 20240315  # Non-interactive"
    echo "  $0 --force                          # Override locks and run"
    echo ""
    exit 0
fi

# Run main function
main "$@"