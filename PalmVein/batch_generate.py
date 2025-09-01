#!/usr/bin/env python3
"""
Batch Palm Vein Image Generator
This script generates palm vein images for multiple configurations at once.
"""

import os
import sys
from main import create_3dtree
from datetime import datetime

def batch_generate_palm_vein_images():
    """
    Generate palm vein images for multiple configurations.
    """
    
    print("🌴 Batch Palm Vein Image Generator")
    print("=" * 50)
    
    # Define configurations to generate
    configurations = [
        {"case": 1, "scale": 0, "samples": 5, "name": "case1_scale0"},
        {"case": 1, "scale": 1, "samples": 5, "name": "case1_scale1"},
        {"case": 2, "scale": 0, "samples": 5, "name": "case2_scale0"},
        {"case": 2, "scale": 1, "samples": 5, "name": "case2_scale1"},
        {"case": 3, "scale": 0, "samples": 5, "name": "case3_scale0"},
        {"case": 3, "scale": 1, "samples": 5, "name": "case3_scale1"},
        {"case": 4, "scale": 0, "samples": 5, "name": "case4_scale0"},
        {"case": 4, "scale": 1, "samples": 5, "name": "case4_scale1"},
    ]
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"batch_generated_{timestamp}"
    
    total_generated = 0
    
    for i, config in enumerate(configurations, 1):
        print(f"\n[{i}/{len(configurations)}] Generating {config['name']}...")
        print(f"   Case: {config['case']}, Scale: {config['scale']}, Samples: {config['samples']}")
        
        # Create output directories
        output_dir = os.path.join(base_output_dir, config['name'])
        full_path = os.path.join(output_dir, "full")
        crop_path = os.path.join(output_dir, "crop")
        
        os.makedirs(full_path, exist_ok=True)
        os.makedirs(crop_path, exist_ok=True)
        
        try:
            # Generate the palm vein images
            create_3dtree(
                case=config['case'],
                fullpath=full_path,
                croppath=crop_path,
                s=config['scale'],
                num_sams=config['samples']
            )
            
            # Count generated files
            full_files = len([f for f in os.listdir(full_path) if f.endswith('.png')])
            crop_files = len([f for f in os.listdir(crop_path) if f.endswith('.png')])
            
            total_generated += full_files + crop_files
            
            print(f"   ✅ Generated {full_files} full + {crop_files} cropped images")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🎉 Batch generation completed!")
    print(f"📁 All images saved in: {base_output_dir}")
    print(f"📊 Total images generated: {total_generated}")
    
    return base_output_dir

def main():
    """Main function to run the batch generator."""
    
    try:
        output_dir = batch_generate_palm_vein_images()
        
        print(f"\n📋 Summary:")
        print(f"   - Generated images for 4 palm cases (1-4)")
        print(f"   - Generated images for 2 scales (0-1)")
        print(f"   - 5 samples per configuration")
        print(f"   - Total: 8 configurations × 5 samples × 2 types = 80 images")
        print(f"\n🔍 You can find all generated images in: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Batch generation cancelled by user.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 