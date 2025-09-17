#ifndef WEBCAM_IPC_APP_H
#define WEBCAM_IPC_APP_H

#include <string>
#include <vector>
#include <atomic>
#include <sys/socket.h>
#include <sys/un.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// IPC Configuration
#define SOCKET_PATH "/tmp/webcam_gui_socket"
#define IPC_BUFFER_SIZE 1024

// IPC message structure
typedef struct {
    int type;
    char data[IPC_BUFFER_SIZE - sizeof(int)];
} IPCMessage;

// IPC message types
#define IPC_MSG_DETECTION 1
#define IPC_MSG_FRAME_PROCESSED 2
#define IPC_MSG_STATUS 3
#define IPC_MSG_ERROR 4

/**
 * @brief Webcam IPC Application Class
 * 
 * This class provides a standalone webcam application with AI face detection
 * capabilities that communicates with the LVGL GUI application via IPC.
 * 
 * Features:
 * - Real-time webcam capture and processing
 * - YOLOv8-based face detection using ONNX Runtime
 * - IPC communication with GUI application
 * - Detection change notification
 * - Simulation mode when camera is unavailable
 */
class WebcamIPCApp {
private:
    // Socket and IPC components
    int m_socket_fd;
    struct sockaddr_un m_server_addr;
    std::atomic<bool> m_running;
    
    // Model configuration
    std::string m_model_path;
    int m_previous_detection_count;
    int m_camera_index;
    
    // ONNX Runtime components
    Ort::Env m_env;
    Ort::Session m_session;
    bool m_model_loaded;
    
    // YOLO parameters
    const int INPUT_WIDTH;
    const int INPUT_HEIGHT;
    const float SCORE_THRESHOLD;
    const float NMS_THRESHOLD;
    const float CONFIDENCE_THRESHOLD;
    
    // Detection tracking
    int m_current_detection_count;
    std::vector<cv::Rect> m_last_detections;

public:
    /**
     * @brief Constructor
     * Initializes the webcam IPC application with default model path
     */
    WebcamIPCApp();
    
    /**
     * @brief Destructor
     * Cleans up resources and closes connections
     */
    ~WebcamIPCApp();
    
    /**
     * @brief Initialize the application
     * @return true if initialization successful, false otherwise
     */
    bool init();
    
    /**
     * @brief Clean up resources
     * Closes socket connections and OpenCV windows
     */
    void cleanup();
    
    /**
     * @brief Send IPC message to GUI application
     * @param type Message type (IPC_MSG_* constants)
     * @param data Optional message data
     * @return true if message sent successfully, false otherwise
     */
    bool send_message(int type, const std::string& data = "");
    
    /**
     * @brief Set the path to the YOLOv8 face detection model
     * @param path Path to the ONNX model file
     */
    void set_model_path(const std::string& path);
    
    /**
     * @brief Set the camera index to use
     * @param index Camera device index
     */
    void set_camera_index(int index);
    
    /**
     * @brief Load the YOLOv8 face detection model
     * @return true if model loaded successfully, false otherwise
     */
    bool load_model();
    
    /**
     * @brief Preprocess input image for YOLOv8 model
     * @param input_image Input image to preprocess
     * @return Preprocessed image blob
     */
    cv::Mat preprocess_image(const cv::Mat& input_image);
    
    /**
     * @brief Postprocess YOLOv8 model output to extract detections
     * @param input_image Original input image for coordinate scaling
     * @param output_data Raw model output data
     * @param output_shape Shape of the model output
     * @return Vector of detected face rectangles
     */
    std::vector<cv::Rect> postprocess_output_dynamic(const cv::Mat& input_image,
                                                    const float* output_data,
                                                    const std::vector<int64_t>& output_shape);
    
    /**
     * @brief Postprocess YOLOv8 model output to extract detections with confidence values
     * @param input_image Original input image for coordinate scaling
     * @param output_data Raw model output data
     * @param output_shape Shape of the model output
     * @param detections Output vector for detected face rectangles
     * @param confidences Output vector for detection confidences
     * @param class_ids Output vector for class IDs
     */
    void postprocess_output_with_confidence(const cv::Mat& input_image,
                                          const float* output_data,
                                          const std::vector<int64_t>& output_shape,
                                          std::vector<cv::Rect>& detections,
                                          std::vector<float>& confidences,
                                          std::vector<int>& class_ids);
    
    /**
     * @brief Detect faces in the given frame
     * @param frame Input frame to process
     * @param detections Output vector for detected face rectangles
     * @param confidences Output vector for detection confidences
     * @param class_ids Output vector for class IDs
     */
    void detect_faces(const cv::Mat& frame, 
                     std::vector<cv::Rect>& detections,
                     std::vector<float>& confidences,
                     std::vector<int>& class_ids);
    
    /**
     * @brief Start the webcam processing loop
     * Main processing function that captures frames and performs face detection
     */
    void start_webcam();
    
    /**
     * @brief Stop the webcam processing
     * Signals the processing loop to stop
     */
    void stop_webcam();
    
    /**
     * @brief Check if the application is currently running
     * @return true if running, false otherwise
     */
    bool is_running() const;
    
    /**
     * @brief Check for detection count changes and send notifications
     * @param current_detections Current frame detections
     */
    void check_and_send_detection_changes(const std::vector<cv::Rect>& current_detections);
    
    /**
     * @brief Run simulation mode when camera is unavailable
     * Provides simulated face detection for testing purposes
     */
    void run_simulation();
    
    /**
     * @brief Run real webcam capture and processing
     * Captures frames from webcam and performs face detection
     */
    void run_webcam();
    
    /**
     * @brief Draw detection rectangles on the frame
     * @param frame Frame to draw on
     * @param detections Vector of detection rectangles
     * @param confidences Vector of confidence scores
     * @param class_ids Vector of class IDs
     */
    void draw_detections(cv::Mat& frame,
                        const std::vector<cv::Rect>& detections,
                        const std::vector<float>& confidences,
                        const std::vector<int>& class_ids);
};

/**
 * @brief Signal handler for graceful shutdown
 * @param sig Signal number
 */
void signal_handler(int sig);

#endif // WEBCAM_IPC_APP_H
