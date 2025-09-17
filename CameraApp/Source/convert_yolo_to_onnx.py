#!/usr/bin/env python3
"""
YOLO to ONNX Converter Script

This script converts YOLO PyTorch (.pt) models to ONNX format for use with ONNX Runtime.
It supports various YOLO models and allows customization of input size and other parameters.

Usage:
    python3 convert_yolo_to_onnx.py [input_model.pt] [output_model.onnx] [options]

Examples:
    python3 convert_yolo_to_onnx.py yolov8n.pt yolov8n.onnx
    python3 convert_yolo_to_onnx.py model.pt face_detection.onnx --img-size 640
    python3 convert_yolo_to_onnx.py --help
"""

import argparse
import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import ultralytics
        print(f"✓ ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("✗ ultralytics not found. Installing...")
        os.system("pip3 install ultralytics")
        try:
            import ultralytics
            print(f"✓ ultralytics installed: {ultralytics.__version__}")
        except ImportError:
            print("✗ Failed to install ultralytics")
            return False
    
    try:
        import onnx
        print(f"✓ onnx version: {onnx.__version__}")
    except ImportError:
        print("✗ onnx not found. Installing...")
        os.system("pip3 install onnx")
        try:
            import onnx
            print(f"✓ onnx installed: {onnx.__version__}")
        except ImportError:
            print("✗ Failed to install onnx")
            return False
    
    return True

def convert_yolo_to_onnx(input_path, output_path, img_size=640, simplify=True, opset=11):
    """
    Convert YOLO PyTorch model to ONNX format.
    
    Args:
        input_path (str): Path to input .pt file
        output_path (str): Path to output .onnx file
        img_size (int): Input image size (width=height)
        simplify (bool): Whether to simplify the ONNX model
        opset (int): ONNX opset version
    """
    try:
        from ultralytics import YOLO
        
        print(f"Loading model from: {input_path}")
        model = YOLO(input_path)
        
        print(f"Converting to ONNX format...")
        print(f"  - Input size: {img_size}x{img_size}")
        print(f"  - Simplify: {simplify}")
        print(f"  - Opset: {opset}")
        
        # Export to ONNX
        success = model.export(
            format='onnx',
            imgsz=img_size,
            simplify=simplify,
            opset=opset
        )
        
        # The export method saves to the same directory as the input file
        # We need to move it to the desired output location
        input_file = Path(input_path)
        default_output = input_file.with_suffix('.onnx')
        
        if default_output.exists():
            if str(default_output) != output_path:
                import shutil
                shutil.move(str(default_output), output_path)
            success = True
        else:
            success = False
        
        if success:
            # Get file size
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"✓ Conversion successful!")
            print(f"  - Output: {output_path}")
            print(f"  - Size: {file_size:.2f} MB")
            
            # Validate ONNX model
            try:
                import onnx
                onnx_model = onnx.load(output_path)
                onnx.checker.check_model(onnx_model)
                print(f"✓ ONNX model validation passed")
            except Exception as e:
                print(f"⚠ ONNX model validation failed: {e}")
            
            return True
        else:
            print(f"✗ Conversion failed")
            return False
            
    except Exception as e:
        print(f"✗ Error during conversion: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO PyTorch models to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 convert_yolo_to_onnx.py yolov8n.pt yolov8n.onnx
  python3 convert_yolo_to_onnx.py model.pt face_detection.onnx --img-size 640
  python3 convert_yolo_to_onnx.py --input yolov8s.pt --output yolov8s.onnx --img-size 416
        """
    )
    
    parser.add_argument('input', nargs='?', help='Input YOLO .pt file path')
    parser.add_argument('output', nargs='?', help='Output ONNX .onnx file path')
    parser.add_argument('--input', '-i', help='Input YOLO .pt file path')
    parser.add_argument('--output', '-o', help='Output ONNX .onnx file path')
    parser.add_argument('--img-size', '-s', type=int, default=640, 
                       help='Input image size (default: 640)')
    parser.add_argument('--no-simplify', action='store_true',
                       help='Disable ONNX model simplification')
    parser.add_argument('--opset', type=int, default=11,
                       help='ONNX opset version (default: 11)')
    parser.add_argument('--check-deps', action='store_true',
                       help='Check and install dependencies')
    
    args = parser.parse_args()
    
    # Handle both positional and named arguments
    input_path = args.input or args.input
    output_path = args.output or args.output
    
    # Check dependencies if requested
    if args.check_deps:
        if not check_dependencies():
            sys.exit(1)
        return
    
    # Validate arguments
    if not input_path:
        print("✗ Error: Input file path is required")
        parser.print_help()
        sys.exit(1)
    
    if not output_path:
        # Generate output path from input path
        input_file = Path(input_path)
        output_path = str(input_file.with_suffix('.onnx'))
        print(f"Output path not specified, using: {output_path}")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"✗ Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Check dependencies
    if not check_dependencies():
        print("✗ Error: Required dependencies not available")
        sys.exit(1)
    
    # Convert model
    print(f"Converting {input_path} to {output_path}")
    success = convert_yolo_to_onnx(
        input_path=input_path,
        output_path=output_path,
        img_size=args.img_size,
        simplify=not args.no_simplify,
        opset=args.opset
    )
    
    if success:
        print(f"\n🎉 Conversion completed successfully!")
        print(f"Model saved to: {output_path}")
    else:
        print(f"\n❌ Conversion failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
