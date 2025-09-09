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
        
        // Extract parameters from fuzzer input
        uint32_t input_dims = (data[offset] % 4) + 1; // 1-4 dimensions
        offset++;
        
        uint32_t axis_dims = (data[offset] % 2) + 1; // 1-2 axis dimensions
        offset++;
        
        bool keep_dims = data[offset] % 2;
        offset++;
        
        uint32_t data_type_idx = data[offset] % 3; // Support float32, int32, int64
        offset++;
        
        tensorflow::DataType dtype;
        switch (data_type_idx) {
            case 0: dtype = tensorflow::DT_FLOAT; break;
            case 1: dtype = tensorflow::DT_INT32; break;
            case 2: dtype = tensorflow::DT_INT64; break;
            default: dtype = tensorflow::DT_FLOAT; break;
        }
        
        // Create input tensor shape
        tensorflow::TensorShape input_shape;
        for (uint32_t i = 0; i < input_dims && offset < size; i++) {
            int64_t dim_size = (data[offset] % 10) + 1; // 1-10 size per dimension
            input_shape.AddDim(dim_size);
            offset++;
        }
        
        if (offset >= size) return 0;
        
        // Create input tensor
        tensorflow::Tensor input_tensor(dtype, input_shape);
        
        // Fill tensor with data from fuzzer input
        int64_t num_elements = input_tensor.NumElements();
        if (dtype == tensorflow::DT_FLOAT) {
            auto flat = input_tensor.flat<float>();
            for (int64_t i = 0; i < num_elements && offset < size; i++) {
                flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
                offset++;
            }
        } else if (dtype == tensorflow::DT_INT32) {
            auto flat = input_tensor.flat<int32_t>();
            for (int64_t i = 0; i < num_elements && offset < size; i++) {
                flat(i) = static_cast<int32_t>(data[offset % size]);
                offset++;
            }
        } else if (dtype == tensorflow::DT_INT64) {
            auto flat = input_tensor.flat<int64_t>();
            for (int64_t i = 0; i < num_elements && offset < size; i++) {
                flat(i) = static_cast<int64_t>(data[offset % size]);
                offset++;
            }
        }
        
        // Create axis tensor
        tensorflow::TensorShape axis_shape({static_cast<int64_t>(axis_dims)});
        tensorflow::Tensor axis_tensor(tensorflow::DT_INT32, axis_shape);
        auto axis_flat = axis_tensor.flat<int32_t>();
        
        int32_t input_rank = input_shape.dims();
        for (uint32_t i = 0; i < axis_dims && offset < size; i++) {
            int32_t axis_val = static_cast<int32_t>(data[offset % size]) % input_rank;
            if (axis_val < 0) axis_val += input_rank;
            axis_flat(i) = axis_val;
            offset++;
        }
        
        // Create TensorFlow scope and session
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create placeholders
        auto input_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto axis_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        
        // Create Sum operation
        auto sum_op = tensorflow::ops::Sum(root, input_placeholder, axis_placeholder,
                                          tensorflow::ops::Sum::KeepDims(keep_dims));
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({{input_placeholder, input_tensor},
                                                 {axis_placeholder, axis_tensor}},
                                                {sum_op}, &outputs);
        
        if (!status.ok()) {
            std::cout << "TensorFlow operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output
        if (!outputs.empty()) {
            const tensorflow::Tensor& result = outputs[0];
            if (result.dtype() != dtype) {
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