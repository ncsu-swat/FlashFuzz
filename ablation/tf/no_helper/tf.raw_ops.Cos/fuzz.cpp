#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/platform/env.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < sizeof(uint32_t) + sizeof(float)) {
            return 0;
        }
        
        // Extract tensor dimensions
        uint32_t num_elements;
        std::memcpy(&num_elements, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        // Limit number of elements to prevent excessive memory usage
        num_elements = num_elements % 1000 + 1;
        
        if (offset + num_elements * sizeof(float) > size) {
            return 0;
        }
        
        // Create TensorFlow scope and session
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input tensor shape
        tensorflow::TensorShape shape({static_cast<int64_t>(num_elements)});
        
        // Create input tensor with float32 data
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, shape);
        auto input_flat = input_tensor.flat<float>();
        
        // Fill tensor with fuzz data
        for (uint32_t i = 0; i < num_elements; ++i) {
            float value;
            if (offset + sizeof(float) <= size) {
                std::memcpy(&value, data + offset, sizeof(float));
                offset += sizeof(float);
            } else {
                value = 0.0f;
            }
            input_flat(i) = value;
        }
        
        // Create placeholder for input
        auto input_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        // Apply Cos operation
        auto cos_op = tensorflow::ops::Cos(root, input_placeholder);
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({{input_placeholder, input_tensor}}, {cos_op}, &outputs);
        
        if (!status.ok()) {
            std::cout << "TensorFlow operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output tensor properties
        if (!outputs.empty()) {
            const tensorflow::Tensor& output = outputs[0];
            if (output.dtype() != tensorflow::DT_FLOAT) {
                std::cout << "Output dtype mismatch" << std::endl;
                return 0;
            }
            if (output.shape() != shape) {
                std::cout << "Output shape mismatch" << std::endl;
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