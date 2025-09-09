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
        
        if (size < sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint8_t)) {
            return 0;
        }
        
        // Extract tensor dimensions
        uint32_t num_elements;
        memcpy(&num_elements, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        // Limit number of elements to prevent excessive memory usage
        num_elements = num_elements % 1000 + 1;
        
        // Extract data type
        uint8_t dtype_val = data[offset] % 10;
        offset += sizeof(uint8_t);
        
        tensorflow::DataType dtype;
        switch (dtype_val) {
            case 0: dtype = tensorflow::DT_FLOAT; break;
            case 1: dtype = tensorflow::DT_DOUBLE; break;
            case 2: dtype = tensorflow::DT_INT32; break;
            case 3: dtype = tensorflow::DT_INT64; break;
            case 4: dtype = tensorflow::DT_HALF; break;
            case 5: dtype = tensorflow::DT_BFLOAT16; break;
            case 6: dtype = tensorflow::DT_INT8; break;
            case 7: dtype = tensorflow::DT_INT16; break;
            case 8: dtype = tensorflow::DT_COMPLEX64; break;
            case 9: dtype = tensorflow::DT_COMPLEX128; break;
            default: dtype = tensorflow::DT_FLOAT; break;
        }
        
        // Create tensor shape
        tensorflow::TensorShape shape({static_cast<int64_t>(num_elements)});
        
        // Create input tensor
        tensorflow::Tensor input_tensor(dtype, shape);
        
        // Fill tensor with fuzz data
        size_t element_size = 0;
        switch (dtype) {
            case tensorflow::DT_FLOAT:
                element_size = sizeof(float);
                break;
            case tensorflow::DT_DOUBLE:
                element_size = sizeof(double);
                break;
            case tensorflow::DT_INT32:
                element_size = sizeof(int32_t);
                break;
            case tensorflow::DT_INT64:
                element_size = sizeof(int64_t);
                break;
            case tensorflow::DT_HALF:
                element_size = sizeof(uint16_t);
                break;
            case tensorflow::DT_BFLOAT16:
                element_size = sizeof(uint16_t);
                break;
            case tensorflow::DT_INT8:
                element_size = sizeof(int8_t);
                break;
            case tensorflow::DT_INT16:
                element_size = sizeof(int16_t);
                break;
            case tensorflow::DT_COMPLEX64:
                element_size = sizeof(std::complex<float>);
                break;
            case tensorflow::DT_COMPLEX128:
                element_size = sizeof(std::complex<double>);
                break;
            default:
                element_size = sizeof(float);
                break;
        }
        
        size_t required_data_size = num_elements * element_size;
        if (offset + required_data_size > size) {
            return 0;
        }
        
        // Copy fuzz data to tensor
        void* tensor_data = input_tensor.data();
        memcpy(tensor_data, data + offset, required_data_size);
        
        // Create TensorFlow scope and session
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create placeholder for input
        auto input_placeholder = tensorflow::ops::Placeholder(root, dtype);
        
        // Create Reciprocal operation
        auto reciprocal_op = tensorflow::ops::Reciprocal(root, input_placeholder);
        
        // Create session
        tensorflow::ClientSession session(root);
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({{input_placeholder, input_tensor}}, {reciprocal_op}, &outputs);
        
        if (!status.ok()) {
            // Operation failed, which is acceptable for fuzzing
            return 0;
        }
        
        // Verify output tensor properties
        if (!outputs.empty()) {
            const tensorflow::Tensor& output = outputs[0];
            if (output.dtype() != dtype || output.shape() != shape) {
                std::cout << "Output tensor properties mismatch" << std::endl;
                return -1;
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}