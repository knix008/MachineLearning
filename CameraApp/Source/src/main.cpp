#include "webcam.h"
#include <iostream>
#include <string>
#include <signal.h>
#include <cstring>
#include <fstream>

// Global variable for signal handling
WebcamIPCApp* g_webcam_app = nullptr;

// Signal handler for graceful shutdown
void signal_handler(int sig) {
    if (g_webcam_app) {
        std::cout << "\nReceived signal " << sig << ", shutting down..." << std::endl;
        g_webcam_app->stop_webcam();
    }
    exit(0);
}

// Function to print usage information
void print_usage(const char* program_name) {
    std::cout << "Webcam Application with AI Face Detection" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Usage: " << program_name << " [OPTIONS]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --model <path>     Path to YOLOv8 face detection model (default: models/yolov8n-face.onnx)" << std::endl;
    std::cout << "  --camera <index>   Camera device index (default: auto-detect)" << std::endl;
    std::cout << "  --confidence <val> Confidence threshold (default: 0.25)" << std::endl;
    std::cout << "  --nms <val>        NMS threshold (default: 0.4)" << std::endl;
    std::cout << "  --width <pixels>   Input width (default: 640)" << std::endl;
    std::cout << "  --height <pixels>  Input height (default: 640)" << std::endl;
    std::cout << "  --simulation       Run in simulation mode (no camera)" << std::endl;
    std::cout << "  --help, -h         Show this help message" << std::endl;
    std::cout << "  --version, -v      Show version information" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << "                                    # Run with default settings" << std::endl;
    std::cout << "  " << program_name << " --model models/yolov8n-face.onnx   # Specify model path" << std::endl;
    std::cout << "  " << program_name << " --camera 0 --confidence 0.5        # Use camera 0 with 50% confidence" << std::endl;
    std::cout << "  " << program_name << " --simulation                       # Run in simulation mode" << std::endl;
    std::cout << std::endl;
}

// Function to print version information
void print_version() {
    std::cout << "Webcam Application v1.0.0" << std::endl;
    std::cout << "AI Face Detection using YOLOv8" << std::endl;
    std::cout << "Built with OpenCV and ONNX Runtime" << std::endl;
    std::cout << std::endl;
}

// Function to parse command line arguments
bool parse_arguments(int argc, char* argv[], std::string& model_path, 
                    int& camera_index, float& confidence, float& nms_threshold,
                    int& width, int& height, bool& simulation_mode) {
    
    // Default values
    model_path = "models/yolov8n-face.onnx";
    camera_index = -1;  // Auto-detect
    confidence = 0.25f;
    nms_threshold = 0.4f;
    width = 640;
    height = 640;
    simulation_mode = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        }
        else if (arg == "--version" || arg == "-v") {
            print_version();
            return false;
        }
        else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        }
        else if (arg == "--camera" && i + 1 < argc) {
            camera_index = std::stoi(argv[++i]);
        }
        else if (arg == "--confidence" && i + 1 < argc) {
            confidence = std::stof(argv[++i]);
        }
        else if (arg == "--nms" && i + 1 < argc) {
            nms_threshold = std::stof(argv[++i]);
        }
        else if (arg == "--width" && i + 1 < argc) {
            width = std::stoi(argv[++i]);
        }
        else if (arg == "--height" && i + 1 < argc) {
            height = std::stoi(argv[++i]);
        }
        else if (arg == "--simulation") {
            simulation_mode = true;
        }
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return false;
        }
    }
    
    return true;
}

// Function to validate parameters
bool validate_parameters(const std::string& model_path, float confidence, 
                        float nms_threshold, int width, int height, bool simulation_mode) {
    
    // Check if model file exists
    if (!simulation_mode) {
        std::ifstream file(model_path);
        if (!file.good()) {
            std::cerr << "Error: Model file not found: " << model_path << std::endl;
            return false;
        }
    }
    
    // Validate confidence threshold
    if (confidence < 0.0f || confidence > 1.0f) {
        std::cerr << "Error: Confidence threshold must be between 0.0 and 1.0" << std::endl;
        return false;
    }
    
    // Validate NMS threshold
    if (nms_threshold < 0.0f || nms_threshold > 1.0f) {
        std::cerr << "Error: NMS threshold must be between 0.0 and 1.0" << std::endl;
        return false;
    }
    
    // Validate dimensions
    if (width <= 0 || height <= 0) {
        std::cerr << "Error: Width and height must be positive" << std::endl;
        return false;
    }
    
    return true;
}

// Main function
int main(int argc, char* argv[]) {
    std::cout << "Starting Webcam Application..." << std::endl;
    
    // Parse command line arguments
    std::string model_path;
    int camera_index;
    float confidence, nms_threshold;
    int width, height;
    bool simulation_mode;
    
    if (!parse_arguments(argc, argv, model_path, camera_index, confidence, 
                        nms_threshold, width, height, simulation_mode)) {
        return 0;
    }
    
    // Validate parameters
    if (!validate_parameters(model_path, confidence, nms_threshold, width, height, simulation_mode)) {
        return 1;
    }
    
    // Create webcam application instance
    WebcamIPCApp webcam_app;
    g_webcam_app = &webcam_app;
    
    // Set up signal handlers for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    try {
        // Initialize the application
        if (!webcam_app.init()) {
            std::cerr << "Failed to initialize webcam application" << std::endl;
            return 1;
        }
        
        // Set model path and camera index
        webcam_app.set_model_path(model_path);
        webcam_app.set_camera_index(camera_index);
        
        // Load the model
        if (!simulation_mode && !webcam_app.load_model()) {
            std::cerr << "Failed to load model: " << model_path << std::endl;
            return 1;
        }
        
        std::cout << "Webcam application started." << std::endl;
        std::cout << "Press ESC or Q to stop the application." << std::endl;
        
        // Start webcam processing
        if (simulation_mode) {
            std::cout << "Running in simulation mode..." << std::endl;
            webcam_app.run_simulation();
        } else {
            webcam_app.start_webcam();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
    
    std::cout << "Webcam Application stopped." << std::endl;
    return 0;
}
