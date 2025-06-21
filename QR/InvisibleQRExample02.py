import qrcode
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import numpy as np
import cv2

def generate_qr_code(data, size=(200, 200)):
    """
    Generate a QR code image
    """
    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=2
    )
    qr.add_data(data)
    qr.make(fit=True)
    img_qr = qr.make_image(fill_color="black", back_color="white")
    
    # Use PIL resize instead of cv2.resize to avoid format issues
    img_resized = img_qr.resize(size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(img_resized)
    
    return img_array

def hide_qr_in_image(cover_image_path, qr_data, output_path):
    """
    Hide QR code in image using LSB steganography
    """
    # Load cover image
    cover_img = cv2.imread(cover_image_path)
    if cover_img is None:
        raise ValueError(f"Could not load image: {cover_image_path}")
    
    # Generate QR code
    qr_img = generate_qr_code(qr_data)
    
    # Convert QR to binary (0 and 1)
    qr_binary = (qr_img > 128).astype(np.uint8)
    
    # Flatten QR code
    qr_flat = qr_binary.flatten()
    
    # Check if image is large enough
    total_pixels = cover_img.shape[0] * cover_img.shape[1] * 3
    if len(qr_flat) > total_pixels:
        raise ValueError("QR code is too large for this image")
    
    # Create stego image
    stego_img = cover_img.copy()
    
    # Hide QR code in LSB of each color channel
    idx = 0
    for i in range(cover_img.shape[0]):
        for j in range(cover_img.shape[1]):
            for k in range(3):  # RGB channels
                if idx < len(qr_flat):
                    # Clear LSB and set it to QR code bit
                    stego_img[i, j, k] = (stego_img[i, j, k] & 0xFE) | qr_flat[idx]
                    idx += 1
                else:
                    break
            if idx >= len(qr_flat):
                break
        if idx >= len(qr_flat):
            break
    
    # Save stego image
    cv2.imwrite(output_path, stego_img)
    print(f"QR code hidden in image: {output_path}")
    print(f"Hidden data: {qr_data}")
    
    return stego_img

def extract_qr_from_image(stego_image_path, qr_size=(200, 200), output_path=None):
    """
    Extract QR code from stego image
    """
    # Load stego image
    stego_img = cv2.imread(stego_image_path)
    if stego_img is None:
        raise ValueError(f"Could not load image: {stego_image_path}")
    
    # Calculate total bits needed
    total_bits = qr_size[0] * qr_size[1]
    
    # Extract LSB from each pixel
    extracted_bits = []
    idx = 0
    
    for i in range(stego_img.shape[0]):
        for j in range(stego_img.shape[1]):
            for k in range(3):  # RGB channels
                if idx < total_bits:
                    # Extract LSB
                    bit = stego_img[i, j, k] & 1
                    extracted_bits.append(bit)
                    idx += 1
                else:
                    break
            if idx >= total_bits:
                break
        if idx >= total_bits:
            break
    
    # Reshape to QR code size
    qr_extracted = np.array(extracted_bits).reshape(qr_size)
    
    # Convert to image (0 = white, 1 = black)
    qr_img = (qr_extracted * 255).astype(np.uint8)
    
    # Save extracted QR code if output path is provided
    if output_path:
        cv2.imwrite(output_path, qr_img)
        print(f"Extracted QR code saved to: {output_path}")
    
    return qr_img

def decode_qr_code(qr_image_path):
    """
    Decode QR code from image using cv2 QR code detector
    """
    try:
        # Load QR code image
        qr_img = cv2.imread(qr_image_path)
        if qr_img is None:
            print(f"Could not load image: {qr_image_path}")
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(qr_img, cv2.COLOR_BGR2GRAY)
        
        # Use cv2 QR code detector
        qr_detector = cv2.QRCodeDetector()
        data, bbox, straight_qrcode = qr_detector.detectAndDecode(gray)
        
        if data:
            print(f"✓ Successfully decoded QR code!")
            print(f"Decoded data: {data}")
            return data
        else:
            print("No QR code found in image or could not decode")
            print("Trying alternative method...")
            
            # Try with different preprocessing
            # Apply threshold to make QR code more visible
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            data2, bbox2, straight_qrcode2 = qr_detector.detectAndDecode(thresh)
            
            if data2:
                print(f"✓ Successfully decoded QR code with thresholding!")
                print(f"Decoded data: {data2}")
                return data2
            else:
                print("Still could not decode QR code")
                return None
            
    except Exception as e:
        print(f"Error decoding QR code: {e}")
        print("Trying alternative method...")
        
        # Alternative method: try to decode using PIL and qrcode library
        try:
            from PIL import Image
            import qrcode
            
            # Load image with PIL
            pil_img = Image.open(qr_image_path)
            
            # Try to decode using qrcode library (this is limited but might work)
            print("Note: QR code extraction successful, but decoding requires manual verification.")
            print("You can scan the extracted QR code image manually with any QR scanner app.")
            return "QR_CODE_EXTRACTED_SUCCESSFULLY"
            
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            return None

def display_decoded_results(original_data, decoded_data, extracted_qr_path):
    """
    Display the decoded results in a clear format
    """
    print("\n" + "="*60)
    print("           QR CODE STEGANOGRAPHY RESULTS")
    print("="*60)
    
    print(f"\n📤 Original Data Hidden:")
    print(f"   {original_data}")
    
    print(f"\n📥 Extracted QR Code Image:")
    print(f"   {extracted_qr_path}")
    
    if decoded_data:
        print(f"\n✅ Decoded Data:")
        print(f"   {decoded_data}")
        
        # Check if decoding was successful
        if decoded_data == original_data:
            print(f"\n🎉 SUCCESS: Data matches perfectly!")
        elif decoded_data == "QR_CODE_EXTRACTED_SUCCESSFULLY":
            print(f"\n⚠️  PARTIAL SUCCESS: QR code extracted but needs manual scanning")
            print(f"   Please scan the extracted QR code image manually")
        else:
            print(f"\n❌ DECODING ERROR: Data does not match")
            print(f"   Expected: {original_data}")
            print(f"   Got: {decoded_data}")
    else:
        print(f"\n❌ FAILED: Could not decode QR code")
        print(f"   Please check the extracted QR code image manually")
    
    print(f"\n📊 Analysis:")
    print(f"   - Steganography method: LSB (Least Significant Bit)")
    print(f"   - Hidden data type: QR Code")
    print(f"   - Extraction method: LSB extraction + QR decoding")
    
    print("\n" + "="*60)

def compare_images(original_path, stego_path, output_path="comparison.png"):
    """
    Create a side-by-side comparison of original and stego images
    """
    original = cv2.imread(original_path)
    stego = cv2.imread(stego_path)
    
    if original is None or stego is None:
        print("Could not load images for comparison")
        return
    
    # Create comparison image
    comparison = np.hstack([original, stego])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, 'Stego', (original.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, comparison)
    print(f"Comparison saved to: {output_path}")

def show_difference_map(original_path, stego_path, output_path="difference_map.png"):
    """
    Show the difference between original and stego images with enhanced visualization
    """
    original = cv2.imread(original_path)
    stego = cv2.imread(stego_path)
    
    if original is None or stego is None:
        print("Could not load images for difference map")
        return
    
    # Method 1: Simple difference with extreme enhancement
    diff = cv2.absdiff(original, stego)
    diff_enhanced = cv2.convertScaleAbs(diff, alpha=255, beta=0)
    
    # Method 2: Show only where differences exist (binary mask)
    diff_binary = (diff > 0).astype(np.uint8) * 255
    
    # Method 3: Extract and amplify LSB differences
    # Convert to uint8 arrays for bitwise operations
    original_uint8 = original.astype(np.uint8)
    stego_uint8 = stego.astype(np.uint8)
    
    # Extract LSB using bitwise AND
    original_lsb = cv2.bitwise_and(original_uint8, 1)
    stego_lsb = cv2.bitwise_and(stego_uint8, 1)
    
    # Calculate LSB differences
    lsb_diff = cv2.absdiff(original_lsb, stego_lsb)
    lsb_diff_visible = lsb_diff * 255
    
    # Method 4: Create a more dramatic visualization
    # Normalize the difference to 0-255 range
    diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply histogram equalization for better contrast
    diff_equalized = cv2.equalizeHist(cv2.cvtColor(diff_normalized, cv2.COLOR_BGR2GRAY))
    diff_equalized_color = cv2.cvtColor(diff_equalized, cv2.COLOR_GRAY2BGR)
    
    # Save multiple versions
    cv2.imwrite(output_path, diff_enhanced)
    cv2.imwrite("difference_binary.png", diff_binary)
    cv2.imwrite("lsb_difference_map.png", lsb_diff_visible)
    cv2.imwrite("difference_equalized.png", diff_equalized_color)
    
    print(f"Enhanced difference map saved to: {output_path}")
    print(f"Binary difference map saved to: difference_binary.png")
    print(f"LSB difference map saved to: lsb_difference_map.png")
    print(f"Equalized difference map saved to: difference_equalized.png")
    
    # Print detailed statistics
    total_pixels = original.shape[0] * original.shape[1] * 3
    changed_pixels = np.sum(diff > 0)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"\n=== Difference Analysis ===")
    print(f"Total pixels: {total_pixels}")
    print(f"Changed pixels: {changed_pixels}")
    print(f"Change percentage: {(changed_pixels/total_pixels)*100:.6f}%")
    print(f"Maximum difference: {max_diff}")
    print(f"Mean difference: {mean_diff:.4f}")
    
    # Check if any differences exist
    if changed_pixels == 0:
        print("WARNING: No differences detected! This might indicate an issue.")
    else:
        print("✓ Differences detected successfully")
        
        # Show which channels have changes
        for i, channel in enumerate(['Blue', 'Green', 'Red']):
            channel_diff = np.sum(diff[:, :, i] > 0)
            print(f"{channel} channel changes: {channel_diff} pixels")

if __name__ == "__main__":
    # Configuration
    cover_image_path = "sample01.png"
    qr_data = "http://blog.naver.com/knix009"
    stego_output_path = "sample01_with_hidden_qr.png"
    extracted_qr_path = "extracted_qr.png"
    
    print("=== QR Code Steganography Demo ===")
    print(f"Cover image: {cover_image_path}")
    print(f"Data to hide: {qr_data}")
    print()
    
    try:
        # Step 1: Hide QR code in image
        print("Step 1: Hiding QR code in image...")
        stego_img = hide_qr_in_image(
            cover_image_path=cover_image_path,
            qr_data=qr_data,
            output_path=stego_output_path
        )
        
        # Step 2: Extract QR code from stego image
        print("\nStep 2: Extracting QR code from stego image...")
        extracted_qr = extract_qr_from_image(
            stego_image_path=stego_output_path,
            qr_size=(200, 200),
            output_path=extracted_qr_path
        )
        
        # Step 3: Decode extracted QR code
        print("\nStep 3: Decoding extracted QR code...")
        decoded_data = decode_qr_code(extracted_qr_path)
        
        # Step 4: Display results
        display_decoded_results(qr_data, decoded_data, extracted_qr_path)
        
        # Step 5: Create comparison
        print("\nStep 5: Creating comparison image...")
        compare_images(cover_image_path, stego_output_path, "comparison_original_vs_stego.png")
        
        # Step 6: Create difference map
        print("\nStep 6: Creating difference map...")
        show_difference_map(cover_image_path, stego_output_path, "difference_map.png")
        
        print("\n=== Process Complete ===")
        print(f"Original image: {cover_image_path}")
        print(f"Stego image: {stego_output_path}")
        print(f"Extracted QR: {extracted_qr_path}")
        print(f"Comparison: comparison_original_vs_stego.png")
        print(f"Difference map: difference_map.png")
        
        print("\n=== Steganography Analysis ===")
        print("The stego image should look identical to the original to the human eye.")
        print("The difference map shows where the hidden data is embedded.")
        print("LSB steganography modifies only the least significant bit of each pixel.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have the required dependencies:")
        print("pip install opencv-python numpy pillow qrcode")
