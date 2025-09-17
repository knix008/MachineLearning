# CameraApp Library Summary

## Quick Reference Table

| Category | Library | Purpose | Size | Source |
|----------|---------|---------|------|---------|
| **AI/ML** | ONNX Runtime 1.16.3 | YOLOv8 inference | ~50MB | Local |
| **Computer Vision** | OpenCV 4.8.1 | Image processing, camera | ~71MB | Local |
| **GUI** | GTK+3 | User interface | ~50MB | System |
| **Multimedia** | GStreamer | Video I/O framework | ~20MB | System |
| **Video Codecs** | FFmpeg | Video/audio processing | ~100MB | System |
| **Graphics** | Cairo | 2D rendering | ~15MB | System |
| **Text** | Pango | Text layout | ~10MB | System |
| **Image Formats** | JPEG/PNG/WebP/TIFF | Image I/O | ~15MB | System |
| **System** | Standard C/C++ | Core functionality | ~50MB | System |

## Direct Dependencies (2 libraries)

1. **ONNX Runtime** - AI model inference
2. **OpenCV** - Computer vision operations

## Transitive Dependencies (~100+ libraries)

### **Essential System Libraries**
- GTK+3 stack (GUI)
- GStreamer (multimedia)
- FFmpeg (codecs)
- X11 (display)
- Standard C/C++ runtime

### **Optional/Feature Libraries**
- Hardware acceleration (VA-API, VDPAU)
- Additional codecs (H.264, H.265, VP9, AV1)
- Security (OpenSSL, GnuTLS)
- Network protocols (SSH, ZeroMQ)

## Total Runtime Memory: ~850MB

- **Application**: 50MB
- **OpenCV**: 100MB  
- **ONNX Runtime**: 200MB
- **System Libraries**: 500MB

## Build vs Runtime Dependencies

| Stage | Dependencies | Count |
|-------|--------------|-------|
| **Build** | Development packages | 8 packages |
| **Runtime** | Shared libraries | 100+ libraries |
| **System** | Services & drivers | OS dependent |

## Architecture Support

- **x86_64**: Full support
- **aarch64/ARM64**: Full support
- **Other**: May require rebuilding

## Distribution Compatibility

- **Ubuntu/Debian**: ✅ Full support
- **CentOS/RHEL/Fedora**: ✅ Full support
- **Arch Linux**: ✅ Full support
- **Other**: ⚠️ May need additional packages

---

**For detailed analysis, see [LIBRARY_ANALYSIS.md](LIBRARY_ANALYSIS.md)**
