# Palm Vein Image Generation - Usage Guide

This guide will help you create palm vein images using the PVTree palm vein generation system.

## 🚀 Quick Start

### Method 1: Interactive Web Interface (Recommended for beginners)
```bash
python gradio_palmvein.py
```
This will open a web interface where you can:
- Select palm case (1-4)
- Adjust scale (1-5)
- Set number of samples (1-10)
- Generate and preview images instantly

### Method 2: Simple Command Line Generator
```bash
python generate_palmvein.py
```
This interactive script will prompt you for:
- Palm case (1-4)
- Scale (0-5)
- Number of samples (1-10)
- Output directory

### Method 3: Batch Generation
```bash
python batch_generate.py
```
This will generate images for all combinations of:
- 4 palm cases (1-4)
- 2 scales (0-1)
- 5 samples each
- Total: 80 images

### Method 4: Direct Script Execution
```bash
python main.py
```
This runs the original generation with default parameters.

## 📁 Output Structure

Generated images are saved in two formats:

### Full Images (`full/`)
- Complete palm vein patterns
- Larger size, includes full palm area
- Good for training and analysis

### Cropped Images (`crop/`)
- Focused on the main vein area
- Smaller, more concentrated patterns
- Good for recognition tasks

## 🎛️ Parameters Explained

### Palm Cases (1-4)
Different palm shapes and vein patterns:
- **Case 1**: Standard palm with balanced vein distribution
- **Case 2**: Wider palm with more spread veins
- **Case 3**: Narrower palm with concentrated veins
- **Case 4**: Unique palm shape with distinctive patterns

### Scale (0-5)
Controls the size and intensity of vein patterns:
- **Scale 0**: Fine, detailed vein patterns
- **Scale 1**: Medium vein patterns (default)
- **Scale 2-5**: Progressively larger, more prominent veins

### Number of Samples
How many variations to generate for each configuration.

## 📊 Generated Images

The system creates realistic palm vein images that include:
- Main vascular tree structure
- Branching vein patterns
- Natural variations in thickness
- Realistic palm geometry
- Different lighting conditions

## 🔧 Advanced Usage

### Custom Generation Script
You can create your own generation script:

```python
from main import create_3dtree

# Generate custom palm vein images
create_3dtree(
    case=2,           # Palm case (1-4)
    fullpath="./my_full_images",    # Full image output path
    croppath="./my_crop_images",    # Cropped image output path
    s=1,              # Scale (0-5)
    num_sams=10       # Number of samples
)
```

### Bezier Curve Generation
For advanced users, you can generate bezier curves:
```bash
python get_bezier.py
```

## 📈 Use Cases

### Research & Development
- Training palm vein recognition models
- Testing biometric algorithms
- Developing new recognition methods

### Data Augmentation
- Expanding training datasets
- Creating synthetic data for testing
- Balancing dataset distributions

### Education & Demonstration
- Teaching biometric concepts
- Demonstrating palm vein patterns
- Visualizing vascular structures

## 🛠️ Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install missing dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. **Python Version Issues**: The system works with Python 3.10-3.13

3. **Memory Issues**: Reduce the number of samples for large-scale generation

4. **Slow Generation**: The process can take time for large batches

### Performance Tips

- Use GPU acceleration if available
- Generate in smaller batches for large datasets
- Use cropped images for faster processing
- Adjust scale parameters based on your needs

## 📚 Additional Resources

- **Paper**: [PVTree: Realistic and Controllable Palm Vein Generation](https://ojs.aaai.org/index.php/AAAI/article/view/32726)
- **Original Repository**: [PVTree-palm-vein-generation](https://github.com/Sunniva-Shang/PVTree-palm-vein-generation)
- **PCE Model**: [PCE-Palm](https://github.com/Ukuer/PCE-Palm)

## 🤝 Support

If you encounter any issues or have questions:
- Check the troubleshooting section above
- Review the original paper and documentation
- Contact the original authors for technical support

## 📄 License

This project follows the license of the original PVTree implementation. Please refer to the original repository for licensing information.

---

**Happy Palm Vein Generation! 🌴** 