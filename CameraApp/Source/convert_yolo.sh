#!/bin/bash

# YOLO to ONNX Converter Shell Script
# This script converts YOLOv8 PyTorch models to ONNX format for use with ONNX Runtime 1.16.3

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/convert_yolo_to_onnx.py"
MODELS_DIR="$SCRIPT_DIR/models"

# Function to check dependencies
check_dependencies() {
    print_status "Checking conversion dependencies..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "python3 is not installed. Please install Python 3 first."
        exit 1
    fi
    
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 is not installed. Please install pip3 first."
        exit 1
    fi
    
    # Check if required Python packages are installed
    if ! python3 -c "import torch" 2>/dev/null; then
        print_warning "PyTorch not found. Installing PyTorch..."
        pip3 install torch torchvision
    fi
    
    if ! python3 -c "import onnx" 2>/dev/null; then
        print_warning "ONNX not found. Installing ONNX..."
        pip3 install onnx
    fi
    
    if ! python3 -c "import ultralytics" 2>/dev/null; then
        print_warning "Ultralytics not found. Installing Ultralytics..."
        pip3 install ultralytics
    fi
    
    print_success "All Python dependencies are available"
}

# Function to show usage
show_usage() {
    echo "YOLO to ONNX Converter Script"
    echo ""
    echo "Usage: $0 [OPTIONS] [MODEL_PATH]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"  
    echo "  -v, --version  Show version information"
    echo "  -c, --check    Check dependencies only"
    echo "  -f, --force    Force conversion even if output exists"
    echo ""
    echo "Arguments:"
    echo "  MODEL_PATH     Path to YOLOv8 PyTorch model (.pt file)"
    echo "                 If not provided, uses default yolov8n-face.pt"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Convert default model"
    echo "  $0 yolov8n-face.pt                   # Convert specific model"
    echo "  $0 --check                           # Check dependencies only"
    echo "  $0 --force yolov8n-face.pt           # Force conversion"
    echo ""
    echo "Output:"
    echo "  Converted models are saved to: $MODELS_DIR/"
    echo "  Compatible with ONNX Runtime 1.16.3"
}

# Function to show version
show_version() {
    echo "YOLO to ONNX Converter Script v1.0"
    echo "Target ONNX Runtime version: 1.16.3"
    echo "Supported models: YOLOv8 (PyTorch format)"
}

# Function to check dependencies only
check_only() {
    check_dependencies
    print_success "Dependency check completed"
}

# Main conversion function
convert_model() {
    local model_path="$1"
    local force_conversion="$2"
    
    # Check if Python script exists
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        print_error "Python conversion script not found: $PYTHON_SCRIPT"
        exit 1
    fi
    
    # Make sure Python script is executable
    chmod +x "$PYTHON_SCRIPT"
    
    # Create models directory if it doesn't exist
    mkdir -p "$MODELS_DIR"
    
    # Check dependencies
    check_dependencies
    
    # Determine model path
    if [ -z "$model_path" ]; then
        model_path="yolov8n-face.pt"
        print_status "Using default model: $model_path"
    fi
    
    # Check if input model exists
    if [ ! -f "$model_path" ]; then
        print_error "Model file not found: $model_path"
        print_status "Please provide a valid YOLOv8 PyTorch model (.pt file)"
        exit 1
    fi
    
    # Determine output path
    local model_name=$(basename "$model_path" .pt)
    local output_path="$MODELS_DIR/${model_name}.onnx"
    
    # Check if output already exists
    if [ -f "$output_path" ] && [ "$force_conversion" != "true" ]; then
        print_warning "Output file already exists: $output_path"
        read -p "Do you want to overwrite it? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Conversion cancelled."
            exit 0
        fi
    fi
    
    print_status "Converting $model_path to ONNX format..."
    print_status "Output will be saved to: $output_path"
    
    # Run the conversion
    if python3 "$PYTHON_SCRIPT" "$model_path" "$output_path"; then
        print_success "Conversion completed successfully!"
        print_status "Model saved to: $output_path"
        print_status "You can now use this model with ONNX Runtime 1.16.3"
    else
        print_error "Conversion failed!"
        exit 1
    fi
}

# Parse command line arguments
CHECK_ONLY=false
FORCE_CONVERSION=false
MODEL_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -v|--version)
            show_version
            exit 0
            ;;
        -c|--check)
            CHECK_ONLY=true
            shift
            ;;
        -f|--force)
            FORCE_CONVERSION=true
            shift
            ;;
        -*)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            if [ -z "$MODEL_PATH" ]; then
                MODEL_PATH="$1"
            else
                print_error "Multiple model paths specified. Please provide only one."
                exit 1
            fi
            shift
            ;;
    esac
done

# Main execution
if [ "$CHECK_ONLY" = true ]; then
    check_only
else
    convert_model "$MODEL_PATH" "$FORCE_CONVERSION"
fi
