#!/bin/bash

# Horse Management App Runner (macOS/Linux)
# 
# This script automatically detects and activates the Python virtual environment,
# then runs the Streamlit-based horse management application.
#
# Usage:
#   ./manage_horses.sh [streamlit-options]

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
    if ! python3 -c "import streamlit, pandas, yaml" 2>/dev/null && ! python -c "import streamlit, pandas, yaml" 2>/dev/null; then
        echo "❌ Required Python packages not found."
        echo ""
        echo "Please either:"
        echo "1. Create and activate a virtual environment with required packages"
        echo "2. Install required packages globally: pip install streamlit pandas pyyaml pillow"
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
    echo "🐴 Horse Management App Runner (macOS/Linux)"
    echo "============================================"
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Activate virtual environment if available
    activate_venv
    
    # Determine Python command
    PYTHON_CMD=$(get_python_cmd)
    
    # Check if manage_horses.py exists
    if [ ! -f "manage_horses.py" ]; then
        echo "❌ manage_horses.py not found in $SCRIPT_DIR"
        echo "Please ensure you're running this script from the horse-id directory."
        exit 1
    fi
    
    # Check if streamlit is available
    if ! $PYTHON_CMD -c "import streamlit" 2>/dev/null; then
        echo "❌ Streamlit not found. Please install it:"
        echo "   pip install streamlit"
        exit 1
    fi
    
    echo ""
    echo "🚀 Starting horse management application..."
    echo "📊 This will open a web interface in your browser"
    echo ""
    
    # Execute Streamlit with manage_horses.py and any additional arguments
    exec $PYTHON_CMD -m streamlit run manage_horses.py "$@"
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n\n🛑 Application stopped."; exit 130' INT

# Show help if requested
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "🐴 Horse Management App Runner (macOS/Linux)"
    echo ""
    echo "Usage:"
    echo "  $0 [streamlit-options]"
    echo ""
    echo "Description:"
    echo "  Automatically activates the Python virtual environment and starts"
    echo "  the Streamlit-based horse management application."
    echo ""
    echo "Examples:"
    echo "  $0                          # Start with default settings"
    echo "  $0 --server.port 8502       # Start on custom port"
    echo "  $0 --server.headless true   # Start in headless mode"
    echo ""
    echo "Features:"
    echo "  - Multi-user safety with pipeline lock detection"
    echo "  - Visual horse image management and review"
    echo "  - Status management (Active, EXCLUDE, REVIEW)"
    echo "  - Canonical ID assignment and merging"
    echo "  - Horse name normalization"
    echo "  - Detection status override"
    echo ""
    echo "The application will open in your default web browser."
    echo "Use Ctrl+C to stop the application."
    echo ""
    exit 0
fi

# Run main function
main "$@"