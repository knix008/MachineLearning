#!/usr/bin/env python3
"""
Simple Palm Vein Image Generator
This script provides an easy way to generate palm vein images with customizable parameters.
"""

import os
import sys
from main import create_3dtree

def generate_palm_vein_images(case=1, scale=1, num_samples=5, output_dir="generated_palmvein"):
    """
    Generate palm vein images with specified parameters.
    
    Args:
        case (int): Palm case (1-4). Different cases represent different palm shapes.
        scale (int): Scale factor (0-5). Controls the size of the vein patterns.
        num_samples (int): Number of samples to generate.
        output_dir (str): Output directory for generated images.
    """
    
    print(f"Generating palm vein images...")
    print(f"Case: {case}, Scale: {scale}, Samples: {num_samples}")
    
    # Create output directories
    full_path = os.path.join(output_dir, "full")
    crop_path = os.path.join(output_dir, "crop")
    
    os.makedirs(full_path, exist_ok=True)
    os.makedirs(crop_path, exist_ok=True)
    
    try:
        # Generate the palm vein images
        create_3dtree(
            case=case,
            fullpath=full_path,
            croppath=crop_path,
            s=scale,
            num_sams=num_samples
        )
        
        print(f"✅ Successfully generated {num_samples} palm vein images!")
        print(f"📁 Full images saved in: {full_path}")
        print(f"📁 Cropped images saved in: {crop_path}")
        
        # List generated files
        full_files = [f for f in os.listdir(full_path) if f.endswith('.png')]
        crop_files = [f for f in os.listdir(crop_path) if f.endswith('.png')]
        
        print(f"\n📊 Generated {len(full_files)} full images and {len(crop_files)} cropped images")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating palm vein images: {e}")
        return False

def main():
    """Main function to run the palm vein generator."""
    
    print("🌴 Palm Vein Image Generator")
    print("=" * 40)
    
    # Get user input
    try:
        case = int(input("Enter palm case (1-4, default=1): ") or "1")
        scale = int(input("Enter scale (0-5, default=1): ") or "1")
        num_samples = int(input("Enter number of samples (1-10, default=5): ") or "5")
        output_dir = input("Enter output directory (default=generated_palmvein): ") or "generated_palmvein"
        
        # Validate inputs
        if not (1 <= case <= 4):
            print("❌ Invalid case. Must be between 1-4.")
            return
            
        if not (0 <= scale <= 5):
            print("❌ Invalid scale. Must be between 0-5.")
            return
            
        if not (1 <= num_samples <= 10):
            print("❌ Invalid number of samples. Must be between 1-10.")
            return
        
        # Generate images
        success = generate_palm_vein_images(case, scale, num_samples, output_dir)
        
        if success:
            print("\n🎉 Palm vein generation completed successfully!")
            print("You can now use these images for your palm vein recognition tasks.")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Generation cancelled by user.")
    except ValueError:
        print("❌ Invalid input. Please enter valid numbers.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 