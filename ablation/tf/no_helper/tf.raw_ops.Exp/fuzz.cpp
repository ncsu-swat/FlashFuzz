#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/framework/types.pb.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint8_t)) {
            return 0;
        }
        
        // Extract data type
        uint8_t dtype_val = data[offset++];
        tensorflow::DataType dtype;
        switch (dtype_val % 6) {
            case 0: dtype = tensorflow::DT_BFLOAT16; break;
            case 1: dtype = tensorflow::DT_HALF; break;
            case 2: dtype = tensorflow::DT_FLOAT; break;
            case 3: dtype = tensorflow::DT_DOUBLE; break;
            case 4: dtype = tensorflow::DT_COMPLEX64; break;
            case 5: dtype = tensorflow::DT_COMPLEX128; break;
            default: dtype = tensorflow::DT_FLOAT; break;
        }
        
        // Extract tensor dimensions
        uint32_t num_dims = (*(uint32_t*)(data + offset)) % 4 + 1;
        offset += sizeof(uint32_t);
        
        if (offset + num_dims * sizeof(uint32_t) > size) {
            return 0;
        }
        
        std::vector<int64_t> dims;
        size_t total_elements = 1;
        for (uint32_t i = 0; i < num_dims; i++) {
            uint32_t dim_size = (*(uint32_t*)(data + offset)) % 10 + 1;
            dims.push_back(dim_size);
            total_elements *= dim_size;
            offset += sizeof(uint32_t);
        }
        
        if (total_elements > 1000) {
            total_elements = 1000;
            dims = {10, 10};
        }
        
        tensorflow::TensorShape shape(dims);
        tensorflow::Tensor input_tensor(dtype, shape);
        
        // Fill tensor with data
        size_t element_size = 0;
        switch (dtype) {
            case tensorflow::DT_BFLOAT16:
                element_size = sizeof(tensorflow::bfloat16);
                break;
            case tensorflow::DT_HALF:
                element_size = sizeof(Eigen::half);
                break;
            case tensorflow::DT_FLOAT:
                element_size = sizeof(float);
                break;
            case tensorflow::DT_DOUBLE:
                element_size = sizeof(double);
                break;
            case tensorflow::DT_COMPLEX64:
                element_size = sizeof(tensorflow::complex64);
                break;
            case tensorflow::DT_COMPLEX128:
                element_size = sizeof(tensorflow::complex128);
                break;
        }
        
        size_t required_data = total_elements * element_size;
        if (offset + required_data > size) {
            // Fill with pattern if not enough data
            auto flat = input_tensor.flat<float>();
            for (int i = 0; i < total_elements; i++) {
                flat(i) = (i % 256) / 128.0f - 1.0f;
            }
        } else {
            // Copy data from fuzz input
            switch (dtype) {
                case tensorflow::DT_FLOAT: {
                    auto flat = input_tensor.flat<float>();
                    for (size_t i = 0; i < total_elements && offset + sizeof(float) <= size; i++) {
                        memcpy(&flat(i), data + offset, sizeof(float));
                        offset += sizeof(float);
                    }
                    break;
                }
                case tensorflow::DT_DOUBLE: {
                    auto flat = input_tensor.flat<double>();
                    for (size_t i = 0; i < total_elements && offset + sizeof(double) <= size; i++) {
                        memcpy(&flat(i), data + offset, sizeof(double));
                        offset += sizeof(double);
                    }
                    break;
                }
                case tensorflow::DT_COMPLEX64: {
                    auto flat = input_tensor.flat<tensorflow::complex64>();
                    for (size_t i = 0; i < total_elements && offset + sizeof(tensorflow::complex64) <= size; i++) {
                        memcpy(&flat(i), data + offset, sizeof(tensorflow::complex64));
                        offset += sizeof(tensorflow::complex64);
                    }
                    break;
                }
                case tensorflow::DT_COMPLEX128: {
                    auto flat = input_tensor.flat<tensorflow::complex128>();
                    for (size_t i = 0; i < total_elements && offset + sizeof(tensorflow::complex128) <= size; i++) {
                        memcpy(&flat(i), data + offset, sizeof(tensorflow::complex128));
                        offset += sizeof(tensorflow::complex128);
                    }
                    break;
                }
                default: {
                    auto flat = input_tensor.flat<float>();
                    for (size_t i = 0; i < total_elements; i++) {
                        flat(i) = (i % 256) / 128.0f - 1.0f;
                    }
                    break;
                }
            }
        }
        
        // Create TensorFlow scope and session
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create placeholder for input
        auto input_placeholder = tensorflow::ops::Placeholder(root, dtype);
        
        // Apply Exp operation
        auto exp_op = tensorflow::ops::Exp(root, input_placeholder);
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({{input_placeholder, input_tensor}}, {exp_op}, &outputs);
        
        if (!status.ok()) {
            std::cout << "TensorFlow operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output tensor properties
        if (!outputs.empty()) {
            const tensorflow::Tensor& output = outputs[0];
            if (output.dtype() != dtype) {
                std::cout << "Output dtype mismatch" << std::endl;
            }
            if (output.shape() != shape) {
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