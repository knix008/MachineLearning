#include "webcam.h"
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <chrono>

// Implementation of WebcamIPCApp class
WebcamIPCApp::WebcamIPCApp() 
    : m_socket_fd(-1), 
      m_running(false),
      m_previous_detection_count(-1),
      m_session(nullptr),
      m_model_loaded(false),
      INPUT_WIDTH(640),
      INPUT_HEIGHT(640),
      SCORE_THRESHOLD(0.25f),  // Increased from 0.1f to reduce false positives
      NMS_THRESHOLD(0.4f),
      CONFIDENCE_THRESHOLD(0.25f),  // Increased from 0.1f to reduce false positives
      m_current_detection_count(0) {
    m_model_path = "../models/yolov8n-face.onnx";
}
    
WebcamIPCApp::~WebcamIPCApp() {
    cleanup();
}
    
bool WebcamIPCApp::init() {
        // Create socket
        m_socket_fd = socket(AF_UNIX, SOCK_DGRAM, 0);
        if (m_socket_fd == -1) {
            std::cerr << "Failed to create socket: " << strerror(errno) << std::endl;
            return false;
        }
        
        // Set up server address
        memset(&m_server_addr, 0, sizeof(m_server_addr));
        m_server_addr.sun_family = AF_UNIX;
        strncpy(m_server_addr.sun_path, SOCKET_PATH, sizeof(m_server_addr.sun_path) - 1);
        
        std::cout << "Webcam IPC application initialized" << std::endl;
        return true;
    }
    
void WebcamIPCApp::cleanup() {
        if (m_socket_fd != -1) {
            close(m_socket_fd);
            m_socket_fd = -1;
        }
        // Close all OpenCV windows
        cv::destroyAllWindows();
    }
    
bool WebcamIPCApp::send_message(int type, const std::string& data) {
        if (m_socket_fd == -1) return false;
        
        IPCMessage msg;
        msg.type = type;
        strncpy(msg.data, data.c_str(), sizeof(msg.data) - 1);
        msg.data[sizeof(msg.data) - 1] = '\0';
        
        ssize_t sent = sendto(m_socket_fd, &msg, sizeof(msg), 0,
                             (struct sockaddr*)&m_server_addr, sizeof(m_server_addr));
        
        if (sent == -1) {
            std::cerr << "Failed to send message: " << strerror(errno) << std::endl;
            return false;
        }
        
        return true;
    }
    
void WebcamIPCApp::set_model_path(const std::string& path) {
        m_model_path = path;
    }
    
void WebcamIPCApp::set_camera_index(int index) {
        m_camera_index = index;
    }
    
bool WebcamIPCApp::load_model() {
        try {
            // Initialize ONNX Runtime environment
            m_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8_Face_Detection");
            
            // Session options
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            
            // Load the model
            m_session = Ort::Session(m_env, m_model_path.c_str(), session_options);
            m_model_loaded = true;
            
            std::cout << "YOLOv8 face detection model loaded successfully" << std::endl;
            return true;
        } catch (const Ort::Exception& e) {
            std::cerr << "Failed to load model: " << e.what() << std::endl;
            return false;
        }
    }
    
cv::Mat WebcamIPCApp::preprocess_image(const cv::Mat& input_image) {
        cv::Mat resized;
        cv::resize(input_image, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
        
        // Convert to float and normalize
        cv::Mat float_img;
        resized.convertTo(float_img, CV_32F, 1.0/255.0);
        
        // Convert BGR to RGB
        cv::cvtColor(float_img, float_img, cv::COLOR_BGR2RGB);
        
        // Create blob
        cv::Mat blob = cv::dnn::blobFromImage(float_img, 1.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
        
        return blob;
    }
    
std::vector<cv::Rect> WebcamIPCApp::postprocess_output_dynamic(const cv::Mat& input_image,
                                                     const float* output_data,
                                                     const std::vector<int64_t>& output_shape) {
        std::vector<cv::Rect> detections;
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        if (output_data == nullptr || output_shape.empty()) {
            return detections;
        }

        // Expect shape like [1, A, C] or [1, C, A]
        int64_t dim1 = output_shape.size() > 1 ? output_shape[1] : 0;
        int64_t dim2 = output_shape.size() > 2 ? output_shape[2] : 0;

        int rows = 0;      // number of anchors
        int dims = 0;      // values per anchor (bbox + conf + classes)
        bool channels_first = false; // if true: [C, A], else [A, C]

        if (output_shape.size() == 3 && dim1 > 0 && dim2 > 0) {
            // For YOLOv8, the format is typically [batch, channels, anchors]
            // where channels = 5 (x, y, w, h, confidence) for single class
            if (dim1 == 5) {
                dims = static_cast<int>(dim1);  // 5 channels
                rows = static_cast<int>(dim2);  // 8400 anchors
                channels_first = true; // layout [C, A] - channels first
            } else {
                rows = static_cast<int>(dim1);
                dims = static_cast<int>(dim2);
                channels_first = false; // layout [A, C]
            }
        } else if (output_shape.size() == 2) {
            rows = static_cast<int>(output_shape[0]);
            dims = static_cast<int>(output_shape[1]);
            channels_first = false;
        } else {
            // Unknown layout
            return detections;
        }

        if (rows <= 0 || dims < 5) return detections;

        /* std::cout << "Processing " << rows << " rows with " << dims << " dimensions" << std::endl; */

        auto value_at = [&](int anchor_idx, int c) -> float {
            if (channels_first) {
                // [C, A]
                return output_data[c * rows + anchor_idx];
            } else {
                // [A, C]
                return output_data[anchor_idx * dims + c];
            }
        };

        int img_w = input_image.cols;
        int img_h = input_image.rows;

        for (int i = 0; i < rows; ++i) {
            float x = value_at(i, 0);
            float y = value_at(i, 1);
            float w = value_at(i, 2);
            float h = value_at(i, 3);
            float conf = value_at(i, 4);

            // Debug: Print first few raw values to understand the format
            /*
            if (i < 3 && conf > 0.001f) {
                std::cout << "Raw values [" << i << "]: x=" << x << ", y=" << y 
                         << ", w=" << w << ", h=" << h << ", conf=" << conf << std::endl;
            }

            // Debug: Print first few confidence values
            if (i < 5) {
                std::cout << "Row " << i << " confidence: " << conf << std::endl;
            }
            */

            if (conf < CONFIDENCE_THRESHOLD) continue;

            int class_id = 0;
            float class_score = 1.0f;

            if (dims > 5) {
                // There are class scores
                int num_classes = dims - 5;
                float best_score = -1.0f;
                int best_id = 0;
                for (int c = 0; c < num_classes; ++c) {
                    float sc = value_at(i, 5 + c);
                    if (sc > best_score) {
                        best_score = sc;
                        best_id = c;
                    }
                }
                if (best_score < SCORE_THRESHOLD) continue;
                class_id = best_id;
                class_score = best_score;
            }

            // Coordinates are in model input space (640x640), need to scale to image space
            float scale_x = static_cast<float>(img_w) / INPUT_WIDTH;
            float scale_y = static_cast<float>(img_h) / INPUT_HEIGHT;
            
            int left = static_cast<int>((x - 0.5f * w) * scale_x);
            int top = static_cast<int>((y - 0.5f * h) * scale_y);
            int width = static_cast<int>(w * scale_x);
            int height = static_cast<int>(h * scale_y);

                    // Debug: Print detection details for high confidence detections
        /*
        if (conf > 0.01f) {  // Only print for reasonable confidence
            std::cout << "Detection " << i << ": x=" << x << ", y=" << y 
                     << ", w=" << w << ", h=" << h << ", conf=" << conf
                     << " -> rect(" << left << "," << top << "," << width << "x" << height << ")"
                     << " [aspect=" << (static_cast<float>(width) / height) << "]" << std::endl;
        }
        */

            // Clamp to image bounds
            left = std::max(0, std::min(img_w - 1, left));
            top = std::max(0, std::min(img_h - 1, top));
            width = std::max(0, std::min(img_w - left, width));
            height = std::max(0, std::min(img_h - top, height));

            // Additional filtering to reduce false positives for face detection
            // Check minimum size (face should be reasonably sized)
            if (width < 20 || height < 20) continue;  // Face should be reasonably sized
            
            // Check aspect ratio (face should have reasonable proportions - roughly square)
            float aspect_ratio = static_cast<float>(width) / height;
            if (aspect_ratio < 0.5f || aspect_ratio > 2.0f) continue;  // Face should be roughly square-ish
            
            // Check if detection is too close to image edges (likely false positive)
            int margin = 10;
            if (left < margin || top < margin || 
                (left + width) > (img_w - margin) || 
                (top + height) > (img_h - margin)) {
                // Only allow edge detections if they have very high confidence
                if (conf < 0.6f) continue;
            }

            boxes.emplace_back(left, top, width, height);
            confidences.push_back(conf * class_score);
            class_ids.push_back(class_id);
        }

        // NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
        
        // Apply adaptive confidence filtering based on number of detections
        std::vector<cv::Rect> filtered_detections;
        std::vector<float> filtered_confidences;
        
        for (int idx : indices) {
            if (idx >= 0 && idx < static_cast<int>(boxes.size())) {
                float detection_conf = confidences[idx];
                
                // Adaptive threshold: if we have many detections, be more strict
                float adaptive_threshold = CONFIDENCE_THRESHOLD;
                if (indices.size() > 3) {
                    adaptive_threshold = std::max(0.4f, CONFIDENCE_THRESHOLD);  // 40% if many detections
                } else if (indices.size() > 1) {
                    adaptive_threshold = std::max(0.3f, CONFIDENCE_THRESHOLD);  // 30% if multiple detections
                }
                
                if (detection_conf >= adaptive_threshold) {
                    filtered_detections.push_back(boxes[idx]);
                    filtered_confidences.push_back(detection_conf);
                }
            }
        }
        
        // Use filtered detections
        detections = filtered_detections;
        
        // Debug: Show filtering results
        /*
        if (!detections.empty()) {
            std::cout << "Filtering: " << indices.size() << " NMS detections -> " 
                     << detections.size() << " final detections" << std::endl;
        }
        */

        return detections;
    }
    
void WebcamIPCApp::postprocess_output_with_confidence(const cv::Mat& input_image,
                                             const float* output_data,
                                             const std::vector<int64_t>& output_shape,
                                             std::vector<cv::Rect>& detections,
                                             std::vector<float>& confidences,
                                             std::vector<int>& class_ids) {
        // Clear output vectors
        detections.clear();
        confidences.clear();
        class_ids.clear();
        
        if (output_data == nullptr || output_shape.empty()) {
            return;
        }

        // Expect shape like [1, A, C] or [1, C, A]
        int64_t dim1 = output_shape.size() > 1 ? output_shape[1] : 0;
        int64_t dim2 = output_shape.size() > 2 ? output_shape[2] : 0;

        int rows = 0;      // number of anchors
        int dims = 0;      // values per anchor (bbox + conf + classes)
        bool channels_first = false; // if true: [C, A], else [A, C]

        if (output_shape.size() == 3 && dim1 > 0 && dim2 > 0) {
            // For YOLOv8, the format is typically [batch, channels, anchors]
            // where channels = 5 (x, y, w, h, confidence) for single class
            if (dim1 == 5) {
                dims = static_cast<int>(dim1);  // 5 channels
                rows = static_cast<int>(dim2);  // 8400 anchors
                channels_first = true; // layout [C, A] - channels first
            } else {
                rows = static_cast<int>(dim1);
                dims = static_cast<int>(dim2);
                channels_first = false; // layout [A, C]
            }
        } else if (output_shape.size() == 2) {
            rows = static_cast<int>(output_shape[0]);
            dims = static_cast<int>(output_shape[1]);
            channels_first = false;
        } else {
            // Unknown layout
            return;
        }

        if (rows <= 0 || dims < 5) return;

        auto value_at = [&](int anchor_idx, int c) -> float {
            if (channels_first) {
                // [C, A]
                return output_data[c * rows + anchor_idx];
            } else {
                // [A, C]
                return output_data[anchor_idx * dims + c];
            }
        };

        int img_w = input_image.cols;
        int img_h = input_image.rows;
        std::vector<cv::Rect> boxes;
        std::vector<float> temp_confidences;
        std::vector<int> temp_class_ids;

        for (int i = 0; i < rows; ++i) {
            float x = value_at(i, 0);
            float y = value_at(i, 1);
            float w = value_at(i, 2);
            float h = value_at(i, 3);
            float conf = value_at(i, 4);

            if (conf < CONFIDENCE_THRESHOLD) continue;

            int class_id = 0;
            float class_score = 1.0f;

            if (dims > 5) {
                // There are class scores
                int num_classes = dims - 5;
                float best_score = -1.0f;
                int best_id = 0;
                for (int c = 0; c < num_classes; ++c) {
                    float sc = value_at(i, 5 + c);
                    if (sc > best_score) {
                        best_score = sc;
                        best_id = c;
                    }
                }
                if (best_score < SCORE_THRESHOLD) continue;
                class_id = best_id;
                class_score = best_score;
            }

            // Coordinates are in model input space (640x640), need to scale to image space
            float scale_x = static_cast<float>(img_w) / INPUT_WIDTH;
            float scale_y = static_cast<float>(img_h) / INPUT_HEIGHT;
            
            int left = static_cast<int>((x - 0.5f * w) * scale_x);
            int top = static_cast<int>((y - 0.5f * h) * scale_y);
            int width = static_cast<int>(w * scale_x);
            int height = static_cast<int>(h * scale_y);

            // Clamp to image bounds
            left = std::max(0, std::min(img_w - 1, left));
            top = std::max(0, std::min(img_h - 1, top));
            width = std::max(0, std::min(img_w - left, width));
            height = std::max(0, std::min(img_h - top, height));

            // Additional filtering to reduce false positives for face detection
            // Check minimum size (face should be reasonably sized)
            if (width < 20 || height < 20) continue;  // Face should be reasonably sized
            
            // Check aspect ratio (face should have reasonable proportions - roughly square)
            float aspect_ratio = static_cast<float>(width) / height;
            if (aspect_ratio < 0.5f || aspect_ratio > 2.0f) continue;  // Face should be roughly square-ish
            
            // Check if detection is too close to image edges (likely false positive)
            int margin = 10;
            if (left < margin || top < margin || 
                (left + width) > (img_w - margin) || 
                (top + height) > (img_h - margin)) {
                // Only allow edge detections if they have very high confidence
                if (conf < 0.6f) continue;
            }

            boxes.emplace_back(left, top, width, height);
            temp_confidences.push_back(conf * class_score);
            temp_class_ids.push_back(class_id);
        }

        // NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, temp_confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
        
        // Apply adaptive confidence filtering based on number of detections
        for (int idx : indices) {
            if (idx >= 0 && idx < static_cast<int>(boxes.size())) {
                float detection_conf = temp_confidences[idx];
                
                // Adaptive threshold: if we have many detections, be more strict
                float adaptive_threshold = CONFIDENCE_THRESHOLD;
                if (indices.size() > 3) {
                    adaptive_threshold = std::max(0.4f, CONFIDENCE_THRESHOLD);  // 40% if many detections
                } else if (indices.size() > 1) {
                    adaptive_threshold = std::max(0.3f, CONFIDENCE_THRESHOLD);  // 30% if multiple detections
                }
                
                if (detection_conf >= adaptive_threshold) {
                    detections.push_back(boxes[idx]);
                    confidences.push_back(detection_conf);
                    class_ids.push_back(temp_class_ids[idx]);
                }
            }
        }
    }
    
void WebcamIPCApp::detect_faces(const cv::Mat& frame, std::vector<cv::Rect>& detections, 
                     std::vector<float>& confidences, std::vector<int>& class_ids) {
        if (!m_model_loaded) {
            std::cerr << "Model not loaded" << std::endl;
            return;
        }
        
        // Clear previous detections
        detections.clear();
        confidences.clear();
        class_ids.clear();
        
        try {
            // Preprocess image
            cv::Mat blob = preprocess_image(frame);
            
            // Prepare input tensor
            std::vector<float> input_tensor(blob.begin<float>(), blob.end<float>());
            
            // Define input shape
            std::vector<int64_t> input_shape = {1, 3, INPUT_HEIGHT, INPUT_WIDTH};
            
            // Create input tensor
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
                memory_info, input_tensor.data(), input_tensor.size(),
                input_shape.data(), input_shape.size());
            
            // YOLOv8 standard input/output names
            const char* input_names[] = {"images"};
            const char* output_names[] = {"output0"};
            
            // Run inference
            std::vector<Ort::Value> output_tensors = m_session.Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input_tensor_ort,
                1,
                output_names,
                1);

            if (output_tensors.empty()) {
                std::cerr << "ONNX Runtime returned no outputs" << std::endl;
                return;
            }
            if (!output_tensors[0].IsTensor()) {
                std::cerr << "ONNX Runtime output[0] is not a tensor" << std::endl;
                return;
            }

            // Get output data
            float* output_data = nullptr;
            try {
                output_data = output_tensors[0].GetTensorMutableData<float>();
            } catch (const Ort::Exception& e) {
                std::cerr << "Failed to access tensor data: " << e.what() << std::endl;
                return;
            }
            auto output_type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
            auto output_shape = output_type_info.GetShape();
            size_t output_size = output_type_info.GetElementCount();
            if (output_size == 0 || output_data == nullptr) {
                std::cerr << "ONNX Runtime output tensor is empty" << std::endl;
                return;
            }

            std::vector<float> output_vector(output_data, output_data + output_size);
            
            // Postprocess to get detections with confidence values
            std::vector<cv::Rect> face_detections = postprocess_output_dynamic(frame, output_data, output_shape);
            
            // Debug output
            /*
            std::cout << "Output shape: [";
            for (size_t i = 0; i < output_shape.size(); ++i) {
                std::cout << output_shape[i];
                if (i < output_shape.size() - 1) std::cout << ", ";
            }
            std::cout << "], size: " << output_size << std::endl;
            std::cout << "Detected " << face_detections.size() << " faces" << std::endl;
            */
            
            // Get the actual confidence values from postprocessing
            std::vector<float> actual_confidences;
            std::vector<int> actual_class_ids;
            
            // We need to call postprocess again to get confidence values, or modify postprocess to return them
            // For now, let's modify the approach to get real confidence values
            postprocess_output_with_confidence(frame, output_data, output_shape, 
                                             detections, confidences, class_ids);
            
        } catch (const Ort::Exception& e) {
            std::cerr << "Inference error: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error during face detection: " << e.what() << std::endl;
        }
    }
    
void WebcamIPCApp::start_webcam() {
        if (m_running.exchange(true)) {
            std::cout << "Webcam already running" << std::endl;
            return;
        }
        
        send_message(IPC_MSG_STATUS, "Starting webcam...");
        
        // Load the YOLOv8 face detection model
        if (!load_model()) {
            send_message(IPC_MSG_ERROR, "Failed to load YOLOv8 face detection model");
            std::cerr << "Failed to load model, falling back to simulation mode" << std::endl;
            run_simulation();
            return;
        }
        
        send_message(IPC_MSG_STATUS, "YOLOv8 face detection model loaded successfully");
        
        // Start real webcam capture
        send_message(IPC_MSG_STATUS, "Starting webcam capture...");
        run_webcam();
        return;
    }
    
void WebcamIPCApp::stop_webcam() {
        m_running.store(false);
    }
    
bool WebcamIPCApp::is_running() const {
        return m_running.load();
    }
    
void WebcamIPCApp::check_and_send_detection_changes(const std::vector<cv::Rect>& current_detections) {
        int new_count = static_cast<int>(current_detections.size());
        
        // Check if detection count has changed
        if (new_count != m_current_detection_count) {
            std::cout << "Detection count changed: " << m_current_detection_count << " -> " << new_count << std::endl;
            
            // Create detection message with coordinates
            std::string detection_msg = "Faces: " + std::to_string(new_count);
            
            // Add coordinates for each detection
            for (size_t i = 0; i < current_detections.size(); ++i) {
                const cv::Rect& rect = current_detections[i];
                detection_msg += " | Face" + std::to_string(i + 1) + ": (" + 
                               std::to_string(rect.x) + "," + std::to_string(rect.y) + "," + 
                               std::to_string(rect.width) + "x" + std::to_string(rect.height) + ")";
            }
            
            // Send detection message to GUI
            send_message(IPC_MSG_DETECTION, detection_msg);
            
            // Update tracking variables
            m_current_detection_count = new_count;
            m_last_detections = current_detections;
        }
    }
    
void WebcamIPCApp::run_simulation() {
        int frame_count = 0;
        
        // Create a simulation window
        cv::namedWindow("Simulation Mode", cv::WINDOW_AUTOSIZE);
        cv::setWindowTitle("Simulation Mode", "No Camera Available - Simulation (640x480)");
        
        // Persistent detection storage for simulation
        std::vector<cv::Rect> sim_detections;
        std::vector<float> sim_confidences;
        std::vector<int> sim_class_ids;
        
        while (m_running.load()) {
            // Create a simulated frame at 640x480 resolution
            cv::Mat sim_frame = cv::Mat::zeros(480, 640, CV_8UC3);
            
            frame_count++;
            
            // Try face detection if model is loaded, otherwise show simulation
            if (frame_count % 5 == 0) {
                if (m_model_loaded) {
                    detect_faces(sim_frame, sim_detections, sim_confidences, sim_class_ids);
                } else {
                    // Simple simulation for when model is not available
                    sim_detections.clear();
                    sim_confidences.clear();
                    sim_class_ids.clear();
                    
                    // Add a simple face-like rectangle
                    sim_detections.push_back(cv::Rect(200, 150, 100, 100));
                    sim_confidences.push_back(0.85f);
                    sim_class_ids.push_back(0);
                }
                
                // Check for detection count changes and send to GUI
                check_and_send_detection_changes(sim_detections);
            }
            
            // Draw detections on simulation frame
            draw_detections(sim_frame, sim_detections, sim_confidences, sim_class_ids);
            
            // Add simulation text
            std::string sim_text = "Simulation Mode - No Camera";
            std::string frame_text = "Frame: " + std::to_string(frame_count);
            std::string time_text = "Time: " + std::to_string(frame_count / 10) + "s";
            std::string res_text = "Resolution: 640x480";
            std::string detection_text = "Detections: " + std::to_string(sim_detections.size());
            
            cv::putText(sim_frame, sim_text, cv::Point(150, 200), 
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            cv::putText(sim_frame, frame_text, cv::Point(150, 250), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            cv::putText(sim_frame, time_text, cv::Point(150, 300), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            cv::putText(sim_frame, res_text, cv::Point(150, 350), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            cv::putText(sim_frame, detection_text, cv::Point(150, 400), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            
            // Display the simulation frame
            cv::imshow("Simulation Mode", sim_frame);
            
            if (frame_count % 30 == 0) { // Every 3 seconds
                send_message(IPC_MSG_STATUS, "Simulation: No camera available (640x480)");
                send_message(IPC_MSG_FRAME_PROCESSED, "Simulated frame: " + std::to_string(frame_count));
            }
            
            // Send initial detection state for simulation
            if (frame_count == 1) {
                send_message(IPC_MSG_DETECTION, "Faces: 0");
            }
            
            // Handle key presses
            int key = cv::waitKey(100) & 0xFF;
            if (key == 27 || key == 'q') { // ESC or Q key
                break;
            }
        }
        
        cv::destroyAllWindows();
    }
    
void WebcamIPCApp::run_webcam() {
        // Open webcam
        cv::VideoCapture cap(0); // Try camera 0
        if (!cap.isOpened()) {
            std::cerr << "Failed to open webcam, trying camera 1..." << std::endl;
            cap.open(1); // Try camera 1
        }
        
        if (!cap.isOpened()) {
            std::cerr << "Failed to open any webcam, falling back to simulation mode" << std::endl;
            send_message(IPC_MSG_ERROR, "Failed to open webcam");
            run_simulation();
            return;
        }
        
        // Set camera properties
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);
        
        std::cout << "Webcam opened successfully" << std::endl;
        send_message(IPC_MSG_STATUS, "Webcam opened successfully");
        
        // Create window
        cv::namedWindow("Webcam Face Detection", cv::WINDOW_AUTOSIZE);
        cv::setWindowTitle("Webcam Face Detection", "Real-time Face Detection (640x480)");
        
        cv::Mat frame;
        int frame_count = 0;
        
        // FPS calculation variables
        auto start_time = std::chrono::high_resolution_clock::now();
        auto last_fps_time = start_time;
        int fps_frame_count = 0;
        double current_fps = 0.0;
        
        // Persistent detection storage
        std::vector<cv::Rect> detections;
        std::vector<float> confidences;
        std::vector<int> class_ids;
        
        while (m_running.load()) {
            // Capture frame
            cap >> frame;
            
            if (frame.empty()) {
                std::cerr << "Failed to capture frame" << std::endl;
                continue;
            }
            
            frame_count++;
            fps_frame_count++;
            
            // Calculate FPS every second
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_fps_time).count();
            
            if (elapsed_time >= 1000) { // Update FPS every second
                current_fps = (fps_frame_count * 1000.0) / elapsed_time;
                fps_frame_count = 0;
                last_fps_time = current_time;
            }
            
            // Run face detection every few frames
            if (frame_count % 3 == 0) {
                detect_faces(frame, detections, confidences, class_ids);
                check_and_send_detection_changes(detections);
            }
            
            // Draw detections on frame
            draw_detections(frame, detections, confidences, class_ids);
            
            // Add status text
            std::string frame_text = "Frame: " + std::to_string(frame_count);
            std::string detection_text = "Detections: " + std::to_string(detections.size());
            std::string fps_text = "FPS: " + std::to_string(static_cast<int>(current_fps));
            
            cv::putText(frame, frame_text, cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, detection_text, cv::Point(10, 60), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, fps_text, cv::Point(10, 90), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            
            // Display the frame
            cv::imshow("Webcam Face Detection", frame);
            
            // Send status updates periodically
            if (frame_count % 90 == 0) { // Every 3 seconds at 30 FPS
                send_message(IPC_MSG_STATUS, "Webcam running: " + std::to_string(detections.size()) + " faces detected");
                send_message(IPC_MSG_FRAME_PROCESSED, "Frame: " + std::to_string(frame_count));
            }
            
            // Handle key presses
            int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q') { // ESC or Q key
                break;
            }
        }
        
        // Release resources
        cap.release();
        cv::destroyAllWindows();
    }
    
void WebcamIPCApp::draw_detections(cv::Mat& frame, const std::vector<cv::Rect>& detections,
                        const std::vector<float>& confidences, const std::vector<int>& class_ids) {
        for (size_t i = 0; i < detections.size(); i++) {
            const cv::Rect& rect = detections[i];
            float conf = confidences[i];
            int class_id = class_ids[i];
            
            // Choose color based on class
            cv::Scalar color;
            std::string label;
            
            switch (class_id) {
                case 0: // face
                    color = cv::Scalar(0, 255, 0); // Green
                    label = "Face";
                    break;
                default:
                    color = cv::Scalar(0, 255, 0); // Green for all classes
                    label = "Unknown";
                    break;
            }
            
            // Draw rectangle
            cv::rectangle(frame, rect, color, 2);
            
            // Draw label with confidence (show one decimal place for better precision)
            char confidence_text[32];
            snprintf(confidence_text, sizeof(confidence_text), "%.1f%%", conf * 100.0f);
            std::string text = label + " " + confidence_text;
            cv::putText(frame, text, cv::Point(rect.x, rect.y - 10), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        }
    }


