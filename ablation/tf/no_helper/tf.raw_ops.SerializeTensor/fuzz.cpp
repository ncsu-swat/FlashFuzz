#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 8) return 0;
        
        // Extract tensor type
        uint8_t dtype_val = data[offset++] % 19; // Limit to valid TensorFlow data types
        tensorflow::DataType dtype;
        switch (dtype_val) {
            case 0: dtype = tensorflow::DT_FLOAT; break;
            case 1: dtype = tensorflow::DT_DOUBLE; break;
            case 2: dtype = tensorflow::DT_INT32; break;
            case 3: dtype = tensorflow::DT_UINT8; break;
            case 4: dtype = tensorflow::DT_INT16; break;
            case 5: dtype = tensorflow::DT_INT8; break;
            case 6: dtype = tensorflow::DT_STRING; break;
            case 7: dtype = tensorflow::DT_INT64; break;
            case 8: dtype = tensorflow::DT_BOOL; break;
            case 9: dtype = tensorflow::DT_QINT8; break;
            case 10: dtype = tensorflow::DT_QUINT8; break;
            case 11: dtype = tensorflow::DT_QINT32; break;
            case 12: dtype = tensorflow::DT_BFLOAT16; break;
            case 13: dtype = tensorflow::DT_QINT16; break;
            case 14: dtype = tensorflow::DT_QUINT16; break;
            case 15: dtype = tensorflow::DT_UINT16; break;
            case 16: dtype = tensorflow::DT_UINT32; break;
            case 17: dtype = tensorflow::DT_UINT64; break;
            default: dtype = tensorflow::DT_FLOAT; break;
        }
        
        // Extract number of dimensions (limit to reasonable size)
        if (offset >= size) return 0;
        uint8_t num_dims = (data[offset++] % 4) + 1; // 1-4 dimensions
        
        // Extract dimensions
        std::vector<int64_t> dims;
        for (int i = 0; i < num_dims && offset + 1 < size; i++) {
            uint8_t dim_size = (data[offset++] % 10) + 1; // 1-10 elements per dimension
            dims.push_back(static_cast<int64_t>(dim_size));
        }
        
        if (dims.empty()) {
            dims.push_back(1);
        }
        
        tensorflow::TensorShape shape(dims);
        tensorflow::Tensor tensor(dtype, shape);
        
        // Fill tensor with data from fuzzer input
        size_t tensor_bytes = tensor.TotalBytes();
        if (tensor_bytes > 0 && offset < size) {
            size_t available_bytes = size - offset;
            size_t copy_bytes = std::min(tensor_bytes, available_bytes);
            
            if (dtype == tensorflow::DT_STRING) {
                // Handle string tensors specially
                auto flat = tensor.flat<tensorflow::tstring>();
                for (int64_t i = 0; i < tensor.NumElements() && offset < size; i++) {
                    uint8_t str_len = (offset < size) ? data[offset++] % 32 : 0;
                    std::string str;
                    for (uint8_t j = 0; j < str_len && offset < size; j++) {
                        str += static_cast<char>(data[offset++]);
                    }
                    flat(i) = str;
                }
            } else {
                // For non-string types, copy raw bytes
                std::memcpy(tensor.data(), data + offset, copy_bytes);
            }
        }
        
        // Create TensorFlow session and graph
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create placeholder for input tensor
        auto placeholder = tensorflow::ops::Placeholder(root, dtype);
        
        // Create SerializeTensor operation
        auto serialize_op = tensorflow::ops::SerializeTensor(root, placeholder);
        
        // Create session
        tensorflow::ClientSession session(root);
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({{placeholder, tensor}}, {serialize_op}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Successfully serialized tensor
            const tensorflow::Tensor& serialized = outputs[0];
            if (serialized.dtype() == tensorflow::DT_STRING) {
                // Verify the output is a string tensor
                auto flat = serialized.flat<tensorflow::tstring>();
                if (flat.size() > 0) {
                    // Access the serialized data
                    const std::string& serialized_data = flat(0);
                    // Basic validation - serialized tensor should have some content
                    if (!serialized_data.empty()) {
                        // Success case - do nothing special
                    }
                }
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}