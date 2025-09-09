#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 20) return 0;
        
        // Extract dimensions and parameters from fuzz input
        uint32_t value_dims = (data[offset] % 4) + 1; // 1-4 dimensions
        offset++;
        
        uint32_t dim_sizes[4];
        for (int i = 0; i < value_dims; i++) {
            dim_sizes[i] = (data[offset] % 10) + 1; // 1-10 size per dimension
            offset++;
        }
        
        // Data format selection
        bool use_nchw = data[offset] % 2;
        offset++;
        
        // Data type selection
        tensorflow::DataType dtype;
        switch (data[offset] % 3) {
            case 0: dtype = tensorflow::DT_FLOAT; break;
            case 1: dtype = tensorflow::DT_DOUBLE; break;
            default: dtype = tensorflow::DT_INT32; break;
        }
        offset++;
        
        // Create value tensor shape
        tensorflow::TensorShape value_shape;
        for (int i = 0; i < value_dims; i++) {
            value_shape.AddDim(dim_sizes[i]);
        }
        
        // Determine bias size based on data format
        int64_t bias_size;
        if (use_nchw && value_dims >= 3) {
            bias_size = dim_sizes[1]; // channels dimension
        } else {
            bias_size = dim_sizes[value_dims - 1]; // last dimension
        }
        
        // Create tensors
        tensorflow::Tensor value_tensor(dtype, value_shape);
        tensorflow::Tensor bias_tensor(dtype, tensorflow::TensorShape({bias_size}));
        
        // Fill tensors with fuzz data
        size_t value_elements = value_tensor.NumElements();
        size_t bias_elements = bias_tensor.NumElements();
        
        if (dtype == tensorflow::DT_FLOAT) {
            auto value_flat = value_tensor.flat<float>();
            auto bias_flat = bias_tensor.flat<float>();
            
            for (int i = 0; i < value_elements && offset < size; i++) {
                value_flat(i) = static_cast<float>(data[offset]) / 255.0f;
                offset++;
            }
            
            for (int i = 0; i < bias_elements && offset < size; i++) {
                bias_flat(i) = static_cast<float>(data[offset]) / 255.0f;
                offset++;
            }
        } else if (dtype == tensorflow::DT_DOUBLE) {
            auto value_flat = value_tensor.flat<double>();
            auto bias_flat = bias_tensor.flat<double>();
            
            for (int i = 0; i < value_elements && offset < size; i++) {
                value_flat(i) = static_cast<double>(data[offset]) / 255.0;
                offset++;
            }
            
            for (int i = 0; i < bias_elements && offset < size; i++) {
                bias_flat(i) = static_cast<double>(data[offset]) / 255.0;
                offset++;
            }
        } else { // DT_INT32
            auto value_flat = value_tensor.flat<int32_t>();
            auto bias_flat = bias_tensor.flat<int32_t>();
            
            for (int i = 0; i < value_elements && offset < size; i++) {
                value_flat(i) = static_cast<int32_t>(data[offset]);
                offset++;
            }
            
            for (int i = 0; i < bias_elements && offset < size; i++) {
                bias_flat(i) = static_cast<int32_t>(data[offset]);
                offset++;
            }
        }
        
        // Create TensorFlow scope and session
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input placeholders
        auto value_placeholder = tensorflow::ops::Placeholder(root, dtype);
        auto bias_placeholder = tensorflow::ops::Placeholder(root, dtype);
        
        // Create BiasAdd operation
        std::string data_format = use_nchw ? "NCHW" : "NHWC";
        auto bias_add = tensorflow::ops::BiasAdd(root, value_placeholder, bias_placeholder,
                                               tensorflow::ops::BiasAdd::DataFormat(data_format));
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({{value_placeholder, value_tensor}, 
                                                {bias_placeholder, bias_tensor}}, 
                                               {bias_add}, &outputs);
        
        if (!status.ok()) {
            std::cout << "BiasAdd operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output tensor properties
        if (!outputs.empty()) {
            const tensorflow::Tensor& output = outputs[0];
            if (output.shape() != value_shape) {
                std::cout << "Output shape mismatch" << std::endl;
            }
            if (output.dtype() != dtype) {
                std::cout << "Output dtype mismatch" << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}