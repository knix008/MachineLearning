import qrcode
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import numpy as np

def generate_invisible_qr_enhanced(
    data, 
    background_path, 
    output_path, 
    alpha=0.08, 
    position='center',
    qr_size=None,
    add_border=False
):
    """
    Generate an invisible QR code with enhanced features
    
    Args:
        data (str): Data to encode in QR code
        background_path (str): Path to background image
        output_path (str): Path to save output image
        alpha (float): Transparency level (0.0 to 1.0)
        position (str): Position of QR code ('center', 'top-left', 'top-right', 'bottom-left', 'bottom-right')
        qr_size (tuple): Custom QR code size (width, height)
        add_border (bool): Whether to add a subtle border around QR code
    """
    
    # Step 1: Generate QR code (using same pattern as working example)
    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=2
    )
    qr.add_data(data)
    qr.make(fit=True)
    img_qr = qr.make_image(fill_color="black", back_color="white").convert("RGBA")
    
    # Step 2: Resize QR code if specified
    if qr_size:
        img_qr = img_qr.resize(qr_size, Image.Resampling.LANCZOS)
    
    # Step 3: Make QR code transparent with enhanced algorithm
    datas = img_qr.getdata()
    new_data = []
    
    for item in datas:
        # Enhanced transparency algorithm
        if item[0] > 200 and item[1] > 200 and item[2] > 200:
            # White background - fully transparent
            new_data.append((255, 255, 255, 0))
        else:
            # Black modules - apply alpha with slight variation for better invisibility
            alpha_value = int(255 * alpha * (0.8 + 0.4 * np.random.random()))
            new_data.append((0, 0, 0, alpha_value))
    
    img_qr.putdata(new_data)
    
    # Step 4: Add subtle border if requested
    if add_border:
        border_size = 20
        bordered_qr = Image.new('RGBA', (img_qr.width + 2*border_size, img_qr.height + 2*border_size), (255, 255, 255, 0))
        bordered_qr.paste(img_qr, (border_size, border_size))
        img_qr = bordered_qr
    
    # Step 5: Load and prepare background
    background = Image.open(background_path).convert("RGBA")
    bg_w, bg_h = background.size
    qr_w, qr_h = img_qr.size
    
    # Step 6: Calculate position
    if position == 'center':
        pos = ((bg_w - qr_w) // 2, (bg_h - qr_h) // 2)
    elif position == 'top-left':
        pos = (50, 50)
    elif position == 'top-right':
        pos = (bg_w - qr_w - 50, 50)
    elif position == 'bottom-left':
        pos = (50, bg_h - qr_h - 50)
    elif position == 'bottom-right':
        pos = (bg_w - qr_w - 50, bg_h - qr_h - 50)
    else:
        pos = ((bg_w - qr_w) // 2, (bg_h - qr_h) // 2)
    
    # Step 7: Overlay QR code on background
    background.paste(img_qr, pos, img_qr)
    
    # Step 8: Save the result
    background.save(output_path, 'PNG')
    print(f"Invisible QR code saved to: {output_path}")
    print(f"QR Code data: {data}")
    print(f"Alpha level: {alpha}")
    print(f"Position: {position}")

def generate_multiple_invisible_qr(data, background_path, output_prefix="invisible_qr"):
    """
    Generate multiple invisible QR codes with different settings for testing
    """
    settings = [
        {"alpha": 0.05, "position": "center", "suffix": "_very_invisible"},
        {"alpha": 0.08, "position": "center", "suffix": "_invisible"},
        {"alpha": 0.12, "position": "center", "suffix": "_slightly_visible"},
        {"alpha": 0.08, "position": "top-left", "suffix": "_top_left"},
        {"alpha": 0.08, "position": "bottom-right", "suffix": "_bottom_right"},
    ]
    
    for setting in settings:
        output_path = f"{output_prefix}{setting['suffix']}.png"
        generate_invisible_qr_enhanced(
            data=data,
            background_path=background_path,
            output_path=output_path,
            alpha=setting["alpha"],
            position=setting["position"]
        )

if __name__ == "__main__":
    # Example usage with sample01.png
    data_to_encode = "http://blog.naver.com/knix009"
    
    # Generate a single invisible QR code
    generate_invisible_qr_enhanced(
        data=data_to_encode,
        background_path="sample01.png",
        output_path="invisible_qr_sample01.png",
        alpha=0.08,  # Very subtle - almost invisible
        position='center',
        add_border=False
    )
    
    # Generate multiple versions for testing
    print("\nGenerating multiple versions for testing...")
    generate_multiple_invisible_qr(
        data=data_to_encode,
        background_path="sample01.png",
        output_prefix="sample01_invisible_qr"
    )
    
    print("\nQR code generation complete!")
    print("You can scan the generated images with any QR code scanner.")
    print("The QR codes are designed to be nearly invisible to the human eye.")
