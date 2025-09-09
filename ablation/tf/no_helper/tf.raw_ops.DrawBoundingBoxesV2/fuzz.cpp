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
        
        if (size < 32) return 0;
        
        // Extract dimensions from fuzz data
        uint32_t batch = (data[offset] % 4) + 1;
        offset += 1;
        uint32_t height = (data[offset] % 64) + 8;
        offset += 1;
        uint32_t width = (data[offset] % 64) + 8;
        offset += 1;
        uint32_t depth = (data[offset] % 4) + 1;
        offset += 1;
        uint32_t num_boxes = (data[offset] % 8) + 1;
        offset += 1;
        uint32_t num_colors = (data[offset] % 8) + 1;
        offset += 1;
        
        // Check if we have enough data
        size_t required_size = batch * height * width * depth * 4 + // images (float32)
                              batch * num_boxes * 4 * 4 + // boxes (float32)
                              num_colors * 4 * 4; // colors (float32, RGBA)
        
        if (offset + required_size > size) return 0;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create images tensor
        tensorflow::TensorShape images_shape({static_cast<int64_t>(batch), 
                                            static_cast<int64_t>(height), 
                                            static_cast<int64_t>(width), 
                                            static_cast<int64_t>(depth)});
        tensorflow::Tensor images_tensor(tensorflow::DT_FLOAT, images_shape);
        auto images_flat = images_tensor.flat<float>();
        
        for (int i = 0; i < images_flat.size() && offset + 4 <= size; ++i) {
            float val;
            std::memcpy(&val, data + offset, sizeof(float));
            images_flat(i) = std::max(0.0f, std::min(1.0f, val));
            offset += 4;
        }
        
        // Create boxes tensor
        tensorflow::TensorShape boxes_shape({static_cast<int64_t>(batch), 
                                           static_cast<int64_t>(num_boxes), 
                                           4});
        tensorflow::Tensor boxes_tensor(tensorflow::DT_FLOAT, boxes_shape);
        auto boxes_flat = boxes_tensor.flat<float>();
        
        for (int i = 0; i < boxes_flat.size() && offset + 4 <= size; ++i) {
            float val;
            std::memcpy(&val, data + offset, sizeof(float));
            boxes_flat(i) = std::max(0.0f, std::min(1.0f, val));
            offset += 4;
        }
        
        // Create colors tensor
        tensorflow::TensorShape colors_shape({static_cast<int64_t>(num_colors), 4});
        tensorflow::Tensor colors_tensor(tensorflow::DT_FLOAT, colors_shape);
        auto colors_flat = colors_tensor.flat<float>();
        
        for (int i = 0; i < colors_flat.size() && offset + 4 <= size; ++i) {
            float val;
            std::memcpy(&val, data + offset, sizeof(float));
            colors_flat(i) = std::max(0.0f, std::min(1.0f, val));
            offset += 4;
        }
        
        // Create input placeholders
        auto images_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto boxes_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto colors_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        // Create DrawBoundingBoxesV2 operation
        auto draw_boxes = tensorflow::ops::DrawBoundingBoxesV2(root, 
                                                              images_placeholder, 
                                                              boxes_placeholder, 
                                                              colors_placeholder);
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({{images_placeholder, images_tensor},
                                                {boxes_placeholder, boxes_tensor},
                                                {colors_placeholder, colors_tensor}},
                                               {draw_boxes}, &outputs);
        
        if (!status.ok()) {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output shape matches input images shape
        if (!outputs.empty() && outputs[0].shape() == images_shape) {
            // Success - output has expected shape
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}