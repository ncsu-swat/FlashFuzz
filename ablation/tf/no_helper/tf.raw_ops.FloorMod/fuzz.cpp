#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract tensor dimensions
        uint32_t dim1 = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        uint32_t dim2 = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        
        // Limit dimensions to reasonable size
        dim1 = (dim1 % 10) + 1;
        dim2 = (dim2 % 10) + 1;
        
        // Extract data type selector
        uint8_t dtype_selector = data[offset++];
        
        // Calculate required data size
        size_t tensor_size = dim1 * dim2;
        size_t required_size = offset + tensor_size * 8; // 8 bytes per element (float64)
        
        if (size < required_size) return 0;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Select data type based on selector
        tensorflow::DataType dtype;
        switch (dtype_selector % 4) {
            case 0: dtype = tensorflow::DT_INT32; break;
            case 1: dtype = tensorflow::DT_INT64; break;
            case 2: dtype = tensorflow::DT_FLOAT; break;
            default: dtype = tensorflow::DT_DOUBLE; break;
        }
        
        tensorflow::TensorShape shape({static_cast<int64_t>(dim1), static_cast<int64_t>(dim2)});
        
        // Create input tensors
        tensorflow::Tensor x_tensor(dtype, shape);
        tensorflow::Tensor y_tensor(dtype, shape);
        
        // Fill tensors with fuzz data
        if (dtype == tensorflow::DT_INT32) {
            auto x_flat = x_tensor.flat<int32_t>();
            auto y_flat = y_tensor.flat<int32_t>();
            for (int i = 0; i < tensor_size && offset + 8 <= size; ++i) {
                int32_t x_val = *reinterpret_cast<const int32_t*>(data + offset);
                offset += 4;
                int32_t y_val = *reinterpret_cast<const int32_t*>(data + offset);
                offset += 4;
                // Avoid division by zero
                if (y_val == 0) y_val = 1;
                x_flat(i) = x_val;
                y_flat(i) = y_val;
            }
        } else if (dtype == tensorflow::DT_INT64) {
            auto x_flat = x_tensor.flat<int64_t>();
            auto y_flat = y_tensor.flat<int64_t>();
            for (int i = 0; i < tensor_size && offset + 8 <= size; ++i) {
                int64_t x_val = *reinterpret_cast<const int32_t*>(data + offset);
                offset += 4;
                int64_t y_val = *reinterpret_cast<const int32_t*>(data + offset);
                offset += 4;
                // Avoid division by zero
                if (y_val == 0) y_val = 1;
                x_flat(i) = x_val;
                y_flat(i) = y_val;
            }
        } else if (dtype == tensorflow::DT_FLOAT) {
            auto x_flat = x_tensor.flat<float>();
            auto y_flat = y_tensor.flat<float>();
            for (int i = 0; i < tensor_size && offset + 8 <= size; ++i) {
                float x_val = *reinterpret_cast<const float*>(data + offset);
                offset += 4;
                float y_val = *reinterpret_cast<const float*>(data + offset);
                offset += 4;
                // Avoid division by zero and NaN/inf
                if (y_val == 0.0f || !std::isfinite(y_val)) y_val = 1.0f;
                if (!std::isfinite(x_val)) x_val = 1.0f;
                x_flat(i) = x_val;
                y_flat(i) = y_val;
            }
        } else { // DT_DOUBLE
            auto x_flat = x_tensor.flat<double>();
            auto y_flat = y_tensor.flat<double>();
            for (int i = 0; i < tensor_size && offset + 8 <= size; ++i) {
                double x_val = *reinterpret_cast<const float*>(data + offset);
                offset += 4;
                double y_val = *reinterpret_cast<const float*>(data + offset);
                offset += 4;
                // Avoid division by zero and NaN/inf
                if (y_val == 0.0 || !std::isfinite(y_val)) y_val = 1.0;
                if (!std::isfinite(x_val)) x_val = 1.0;
                x_flat(i) = x_val;
                y_flat(i) = y_val;
            }
        }
        
        // Create placeholder ops
        auto x_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto y_placeholder = tensorflow::ops::Placeholder(root, dtype);
        
        // Create FloorMod operation
        auto floormod_op = tensorflow::ops::FloorMod(root, x_placeholder, y_placeholder);
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({{x_placeholder, x_tensor}, {y_placeholder, y_tensor}}, 
                                               {floormod_op}, &outputs);
        
        if (!status.ok()) {
            return 0; // Ignore TensorFlow errors for fuzzing
        }
        
        // Verify output tensor properties
        if (!outputs.empty()) {
            const auto& result = outputs[0];
            if (result.dtype() != dtype || result.shape() != shape) {
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