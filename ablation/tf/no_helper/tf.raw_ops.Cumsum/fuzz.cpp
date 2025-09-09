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
        
        // Extract parameters from fuzzer input
        uint32_t num_dims = (data[offset] % 4) + 1; // 1-4 dimensions
        offset++;
        
        if (offset + num_dims * 4 + 8 > size) return 0;
        
        // Create tensor shape
        std::vector<int64_t> dims;
        for (uint32_t i = 0; i < num_dims; i++) {
            uint32_t dim_size = 1 + (data[offset] % 10); // 1-10 elements per dim
            dims.push_back(dim_size);
            offset++;
        }
        
        tensorflow::TensorShape shape(dims);
        int64_t total_elements = shape.num_elements();
        
        if (total_elements > 1000) return 0; // Limit size for fuzzing
        
        // Extract axis
        int32_t axis = static_cast<int32_t>(data[offset] % num_dims);
        if (data[offset + 1] & 1) axis = -axis - 1; // Allow negative axis
        offset += 2;
        
        // Extract boolean flags
        bool exclusive = (data[offset] & 1) != 0;
        bool reverse = (data[offset] & 2) != 0;
        offset++;
        
        // Choose data type
        tensorflow::DataType dtype;
        switch (data[offset] % 6) {
            case 0: dtype = tensorflow::DT_FLOAT; break;
            case 1: dtype = tensorflow::DT_DOUBLE; break;
            case 2: dtype = tensorflow::DT_INT32; break;
            case 3: dtype = tensorflow::DT_INT64; break;
            case 4: dtype = tensorflow::DT_INT16; break;
            default: dtype = tensorflow::DT_INT8; break;
        }
        offset++;
        
        // Create input tensor
        tensorflow::Tensor input_tensor(dtype, shape);
        
        // Fill tensor with data from fuzzer input
        size_t bytes_needed = total_elements * tensorflow::DataTypeSize(dtype);
        if (offset + bytes_needed > size) {
            // Fill with pattern if not enough data
            auto flat = input_tensor.flat<float>();
            for (int64_t i = 0; i < total_elements; i++) {
                if (dtype == tensorflow::DT_FLOAT) {
                    input_tensor.flat<float>()(i) = static_cast<float>((i + 1) % 100);
                } else if (dtype == tensorflow::DT_DOUBLE) {
                    input_tensor.flat<double>()(i) = static_cast<double>((i + 1) % 100);
                } else if (dtype == tensorflow::DT_INT32) {
                    input_tensor.flat<int32_t>()(i) = static_cast<int32_t>((i + 1) % 100);
                } else if (dtype == tensorflow::DT_INT64) {
                    input_tensor.flat<int64_t>()(i) = static_cast<int64_t>((i + 1) % 100);
                } else if (dtype == tensorflow::DT_INT16) {
                    input_tensor.flat<int16_t>()(i) = static_cast<int16_t>((i + 1) % 100);
                } else if (dtype == tensorflow::DT_INT8) {
                    input_tensor.flat<int8_t>()(i) = static_cast<int8_t>((i + 1) % 100);
                }
            }
        } else {
            // Use fuzzer data
            std::memcpy(input_tensor.data(), data + offset, bytes_needed);
        }
        
        // Create axis tensor
        tensorflow::Tensor axis_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        axis_tensor.scalar<int32_t>()() = axis;
        
        // Create TensorFlow scope and session
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create placeholders
        auto x_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto axis_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        
        // Create Cumsum operation
        auto cumsum_op = tensorflow::ops::Cumsum(root, x_placeholder, axis_placeholder,
                                                tensorflow::ops::Cumsum::Exclusive(exclusive)
                                                .Reverse(reverse));
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({{x_placeholder, input_tensor}, 
                                                 {axis_placeholder, axis_tensor}}, 
                                                {cumsum_op}, &outputs);
        
        if (!status.ok()) {
            std::cout << "TensorFlow operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output has same shape as input
        if (outputs.size() != 1) {
            std::cout << "Unexpected number of outputs: " << outputs.size() << std::endl;
            return 0;
        }
        
        if (outputs[0].shape() != input_tensor.shape()) {
            std::cout << "Output shape mismatch" << std::endl;
            return 0;
        }
        
        if (outputs[0].dtype() != input_tensor.dtype()) {
            std::cout << "Output dtype mismatch" << std::endl;
            return 0;
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}