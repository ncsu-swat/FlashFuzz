#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions from fuzz data
        uint32_t batch = (data[offset] % 4) + 1;
        offset++;
        uint32_t height = (data[offset] % 64) + 32;
        offset++;
        uint32_t width = (data[offset] % 64) + 32;
        offset++;
        uint32_t depth = (data[offset] % 3) + 1;
        offset++;
        uint32_t num_boxes = (data[offset] % 8) + 1;
        offset++;
        
        // Use remaining data for tensor values
        size_t remaining_size = size - offset;
        if (remaining_size < 4) return 0;
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create images tensor
        tensorflow::TensorShape images_shape({batch, height, width, depth});
        tensorflow::Tensor images_tensor(tensorflow::DT_FLOAT, images_shape);
        auto images_flat = images_tensor.flat<float>();
        
        // Fill images tensor with fuzz data
        size_t images_size = batch * height * width * depth;
        for (size_t i = 0; i < images_size && offset < size; i++) {
            images_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
            offset++;
        }
        
        // Create boxes tensor
        tensorflow::TensorShape boxes_shape({batch, num_boxes, 4});
        tensorflow::Tensor boxes_tensor(tensorflow::DT_FLOAT, boxes_shape);
        auto boxes_flat = boxes_tensor.flat<float>();
        
        // Fill boxes tensor with normalized coordinates [0.0, 1.0]
        size_t boxes_size = batch * num_boxes * 4;
        for (size_t i = 0; i < boxes_size; i++) {
            float val = static_cast<float>(data[(offset + i) % size]) / 255.0f;
            boxes_flat(i) = val;
        }
        
        // Create input placeholders
        auto images_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto boxes_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        // Create DrawBoundingBoxes operation
        auto draw_boxes = tensorflow::ops::DrawBoundingBoxes(root, images_placeholder, boxes_placeholder);
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run(
            {{images_placeholder, images_tensor}, {boxes_placeholder, boxes_tensor}},
            {draw_boxes},
            &outputs
        );
        
        if (!status.ok()) {
            std::cout << "TensorFlow operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output tensor properties
        if (!outputs.empty()) {
            const tensorflow::Tensor& output = outputs[0];
            if (output.shape() != images_shape) {
                std::cout << "Output shape mismatch" << std::endl;
                return 0;
            }
            if (output.dtype() != tensorflow::DT_FLOAT) {
                std::cout << "Output dtype mismatch" << std::endl;
                return 0;
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}