import gradio as gr
from main import create_3dtree
import tempfile
import os
import glob
import random
from datetime import datetime
from data_tools import cross

def generate_palmvein(case, scale, num_samples, output_format, blend_xray, palm_image_path):
    """
    Generate palm vein images with enhanced functionality.
    
    Args:
        case (int): Palm case (1-4)
        scale (int): Scale factor (0-5)
        num_samples (int): Number of samples to generate
        output_format (str): 'crop' or 'full' or 'both'
    """
    
    try:
        # Always create results directory structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("results", f"palmvein_generated_{timestamp}")
        fullpath = os.path.join(output_dir, 'full')
        croppath = os.path.join(output_dir, 'crop')
        
        # Create directories
        os.makedirs(fullpath, exist_ok=True)
        os.makedirs(croppath, exist_ok=True)
        
        # Generate the palm vein images
        create_3dtree(
            case=case, 
            fullpath=fullpath, 
            croppath=croppath, 
            s=scale, 
            num_sams=num_samples
        )
        
        # Collect generated images and optionally blend (x-ray style)
        results = []
        crop_images = glob.glob(os.path.join(croppath, "sample*.png"))
        crop_images.sort()
        full_images = glob.glob(os.path.join(fullpath, "sample*.png"))
        full_images.sort()

        xray_results = []
        if blend_xray:
            xray_dir = os.path.join(output_dir, 'xray')
            xray_crop_dir = os.path.join(xray_dir, 'crop')
            xray_full_dir = os.path.join(xray_dir, 'full')
            os.makedirs(xray_dir, exist_ok=True)
            os.makedirs(xray_crop_dir, exist_ok=True)
            os.makedirs(xray_full_dir, exist_ok=True)

            # Determine palm image to use
            palm_path = None
            if palm_image_path and os.path.exists(palm_image_path):
                palm_path = palm_image_path
            else:
                default_palms = sorted(glob.glob(os.path.join('bezierpalm', '*.png')))
                if default_palms:
                    palm_path = random.choice(default_palms)

            if palm_path:
                if output_format in ['crop', 'both']:
                    for src in crop_images:
                        dst = os.path.join(xray_crop_dir, os.path.basename(src))
                        cross(src, palm_path, dst)
                        xray_results.append(dst)
                if output_format in ['full', 'both']:
                    for src in full_images:
                        dst = os.path.join(xray_full_dir, os.path.basename(src))
                        cross(src, palm_path, dst)
                        xray_results.append(dst)

        # Build gallery results (prefer showing xray if generated)
        if xray_results:
            xray_results.sort()
            results.extend(xray_results[:min(5, len(xray_results))])
        else:
            if output_format in ['crop', 'both']:
                results.extend(crop_images[:min(5, len(crop_images))])
            if output_format in ['full', 'both']:
                results.extend(full_images[:min(5, len(full_images))])
        
        # Return results
        if results:
            status_msg = f"✅ Successfully generated {num_samples} palm vein images!"
            status_msg += f"\n📁 Saved to: {output_dir}"
            if xray_results:
                status_msg += f"\n🩻 X-ray blends saved under: {os.path.join(output_dir, 'xray')}"
            return results, status_msg
        else:
            return None, "❌ No images were generated. Please check the parameters."
            
    except Exception as e:
        return None, f"❌ Error generating images: {str(e)}"

def create_interface():
    """Create the enhanced Gradio interface."""
    
    with gr.Blocks(
        title="Palm Vein Image Generator",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .output-image {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        """
    ) as iface:
        
        gr.Markdown("""
        # 🌴 Palm Vein Image Generator
        
        Generate realistic synthetic palm vein images using the PVTree 3D tree model. 
        This tool creates high-quality palm vein patterns for research, development, and educational purposes.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Parameters")
                
                case = gr.Slider(
                    minimum=1, maximum=4, step=1, value=1, 
                    label="Palm Case", 
                    info="Different palm shapes and vein patterns (1-4)"
                )
                
                scale = gr.Slider(
                    minimum=0, maximum=5, step=1, value=1, 
                    label="Scale Factor", 
                    info="Controls vein thickness and prominence (0-5)"
                )
                
                num_samples = gr.Slider(
                    minimum=1, maximum=10, step=1, value=5, 
                    label="Number of Samples", 
                    info="How many variations to generate (1-10)"
                )
                
                output_format = gr.Radio(
                    choices=["crop", "full", "both"], 
                    value="crop", 
                    label="Output Format",
                    info="Crop: focused vein area, Full: complete palm, Both: both formats"
                )
                
                blend_xray = gr.Checkbox(
                    label="Blend with palm (x-ray style)",
                    value=False,
                    info="Creates an x-ray style composite using a palm image"
                )

                palm_image = gr.Image(
                    label="Palm image (optional)",
                    type="filepath",
                    sources=["upload"],
                    height=160
                )

                generate_btn = gr.Button(
                    "🚀 Generate Palm Vein Images", 
                    variant="primary",
                    size="lg"
                )
                
                # Parameter descriptions
                gr.Markdown("""
                ### 📋 Parameter Guide
                
                **Palm Cases:**
                - **Case 1**: Standard palm with balanced vein distribution
                - **Case 2**: Wider palm with more spread veins  
                - **Case 3**: Narrower palm with concentrated veins
                - **Case 4**: Unique palm shape with distinctive patterns
                
                **Scale Factors:**
                - **0**: Fine, detailed vein patterns
                - **1**: Medium vein patterns (recommended)
                - **2-5**: Progressively larger, more prominent veins
                """)
            
            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Generated Images")
                
                status = gr.Textbox(
                    label="Status", 
                    interactive=False,
                    placeholder="Click 'Generate' to create palm vein images..."
                )
                
                gallery = gr.Gallery(
                    label="Palm Vein Images",
                    show_label=True,
                    elem_id="gallery",
                    columns=3,
                    rows=2,
                    height="auto",
                    object_fit="contain"
                )
                
                # Quick preset buttons
                gr.Markdown("### ⚡ Quick Presets")
                with gr.Row():
                    preset_btn1 = gr.Button("Case 1, Scale 1", size="sm")
                    preset_btn2 = gr.Button("Case 2, Scale 0", size="sm")
                    preset_btn3 = gr.Button("Case 3, Scale 2", size="sm")
                    preset_btn4 = gr.Button("Case 4, Scale 1", size="sm")
        
        # Event handlers
        generate_btn.click(
            fn=generate_palmvein,
            inputs=[case, scale, num_samples, output_format, blend_xray, palm_image],
            outputs=[gallery, status]
        )
        
        # Preset button handlers
        preset_btn1.click(
            fn=lambda: (1, 1, 5, "crop", False, None),
            outputs=[case, scale, num_samples, output_format, blend_xray, palm_image]
        )
        
        preset_btn2.click(
            fn=lambda: (2, 0, 5, "crop", False, None),
            outputs=[case, scale, num_samples, output_format, blend_xray, palm_image]
        )
        
        preset_btn3.click(
            fn=lambda: (3, 2, 5, "crop", False, None),
            outputs=[case, scale, num_samples, output_format, blend_xray, palm_image]
        )
        
        preset_btn4.click(
            fn=lambda: (4, 1, 5, "crop", False, None),
            outputs=[case, scale, num_samples, output_format, blend_xray, palm_image]
        )
        
        # Footer
        gr.Markdown("""
        ---
        **💡 Tips:**
        - Use 'Crop' format for recognition tasks
        - Use 'Full' format for training and analysis  
        - All generated images are automatically saved to the 'results' directory
        - Try different cases and scales for variety
        
        **🔗 Based on:** [PVTree: Realistic and Controllable Palm Vein Generation](https://ojs.aaai.org/index.php/AAAI/article/view/32726)
        """)
    
    return iface

if __name__ == "__main__":
    iface = create_interface()
    iface.launch(
        inbrowser=True
    )
