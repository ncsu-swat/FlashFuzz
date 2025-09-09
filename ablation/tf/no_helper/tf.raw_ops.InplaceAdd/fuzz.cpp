#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions from fuzz data
        uint32_t x_rows = (data[offset] % 10) + 1;
        offset++;
        uint32_t x_cols = (data[offset] % 10) + 1;
        offset++;
        uint32_t i_size = (data[offset] % x_rows) + 1;
        offset++;
        uint32_t v_rows = i_size;
        uint32_t v_cols = x_cols;
        
        if (offset + x_rows * x_cols * sizeof(float) + i_size * sizeof(int32_t) + v_rows * v_cols * sizeof(float) > size) {
            return 0;
        }
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create tensor x
        tensorflow::TensorShape x_shape({static_cast<int64_t>(x_rows), static_cast<int64_t>(x_cols)});
        tensorflow::Tensor x_tensor(tensorflow::DT_FLOAT, x_shape);
        auto x_flat = x_tensor.flat<float>();
        for (int i = 0; i < x_rows * x_cols; i++) {
            if (offset + sizeof(float) <= size) {
                float val;
                memcpy(&val, data + offset, sizeof(float));
                x_flat(i) = val;
                offset += sizeof(float);
            } else {
                x_flat(i) = 0.0f;
            }
        }
        
        // Create tensor i (indices)
        tensorflow::TensorShape i_shape({static_cast<int64_t>(i_size)});
        tensorflow::Tensor i_tensor(tensorflow::DT_INT32, i_shape);
        auto i_flat = i_tensor.flat<int32_t>();
        for (int j = 0; j < i_size; j++) {
            if (offset + sizeof(int32_t) <= size) {
                int32_t val;
                memcpy(&val, data + offset, sizeof(int32_t));
                i_flat(j) = abs(val) % x_rows;  // Ensure valid index
                offset += sizeof(int32_t);
            } else {
                i_flat(j) = j % x_rows;
            }
        }
        
        // Create tensor v
        tensorflow::TensorShape v_shape({static_cast<int64_t>(v_rows), static_cast<int64_t>(v_cols)});
        tensorflow::Tensor v_tensor(tensorflow::DT_FLOAT, v_shape);
        auto v_flat = v_tensor.flat<float>();
        for (int k = 0; k < v_rows * v_cols; k++) {
            if (offset + sizeof(float) <= size) {
                float val;
                memcpy(&val, data + offset, sizeof(float));
                v_flat(k) = val;
                offset += sizeof(float);
            } else {
                v_flat(k) = 0.0f;
            }
        }
        
        // Create placeholder ops
        auto x_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto i_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto v_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        // Create InplaceAdd operation
        auto inplace_add = tensorflow::ops::InplaceAdd(root, x_placeholder, i_placeholder, v_placeholder);
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({{x_placeholder, x_tensor}, 
                                                 {i_placeholder, i_tensor}, 
                                                 {v_placeholder, v_tensor}}, 
                                                {inplace_add}, &outputs);
        
        if (!status.ok()) {
            std::cout << "TensorFlow operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output tensor properties
        if (!outputs.empty()) {
            const tensorflow::Tensor& result = outputs[0];
            if (result.dtype() != tensorflow::DT_FLOAT) {
                std::cout << "Unexpected output dtype" << std::endl;
            }
            if (result.shape() != x_shape) {
                std::cout << "Output shape mismatch" << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}