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
        
        // Need at least basic parameters
        if (size < 32) return 0;
        
        // Extract dimensions from fuzz data
        uint32_t batch = (data[offset] % 4) + 1;
        offset++;
        uint32_t in_height = (data[offset] % 8) + 1;
        offset++;
        uint32_t in_width = (data[offset] % 8) + 1;
        offset++;
        uint32_t depth = (data[offset] % 4) + 1;
        offset++;
        
        uint32_t filter_height = (data[offset] % 5) + 1;
        offset++;
        uint32_t filter_width = (data[offset] % 5) + 1;
        offset++;
        
        // Extract strides and rates
        uint32_t stride_height = (data[offset] % 3) + 1;
        offset++;
        uint32_t stride_width = (data[offset] % 3) + 1;
        offset++;
        uint32_t rate_height = (data[offset] % 3) + 1;
        offset++;
        uint32_t rate_width = (data[offset] % 3) + 1;
        offset++;
        
        // Extract padding type
        bool use_same_padding = (data[offset] % 2) == 0;
        offset++;
        
        // Extract data type
        tensorflow::DataType dtype = tensorflow::DT_FLOAT;
        uint8_t dtype_choice = data[offset] % 3;
        offset++;
        switch (dtype_choice) {
            case 0: dtype = tensorflow::DT_FLOAT; break;
            case 1: dtype = tensorflow::DT_INT32; break;
            case 2: dtype = tensorflow::DT_UINT8; break;
        }
        
        // Create input tensor
        tensorflow::TensorShape input_shape({batch, in_height, in_width, depth});
        tensorflow::Tensor input_tensor(dtype, input_shape);
        
        // Create filter tensor
        tensorflow::TensorShape filter_shape({filter_height, filter_width, depth});
        tensorflow::Tensor filter_tensor(dtype, filter_shape);
        
        // Fill tensors with fuzz data
        size_t input_elements = input_tensor.NumElements();
        size_t filter_elements = filter_tensor.NumElements();
        
        if (dtype == tensorflow::DT_FLOAT) {
            auto input_flat = input_tensor.flat<float>();
            auto filter_flat = filter_tensor.flat<float>();
            
            for (int i = 0; i < input_elements && offset < size; i++, offset++) {
                input_flat(i) = static_cast<float>(data[offset]) / 255.0f;
            }
            for (int i = 0; i < filter_elements && offset < size; i++, offset++) {
                filter_flat(i) = static_cast<float>(data[offset]) / 255.0f;
            }
        } else if (dtype == tensorflow::DT_INT32) {
            auto input_flat = input_tensor.flat<int32_t>();
            auto filter_flat = filter_tensor.flat<int32_t>();
            
            for (int i = 0; i < input_elements && offset < size; i++, offset++) {
                input_flat(i) = static_cast<int32_t>(data[offset]);
            }
            for (int i = 0; i < filter_elements && offset < size; i++, offset++) {
                filter_flat(i) = static_cast<int32_t>(data[offset]);
            }
        } else if (dtype == tensorflow::DT_UINT8) {
            auto input_flat = input_tensor.flat<uint8_t>();
            auto filter_flat = filter_tensor.flat<uint8_t>();
            
            for (int i = 0; i < input_elements && offset < size; i++, offset++) {
                input_flat(i) = data[offset];
            }
            for (int i = 0; i < filter_elements && offset < size; i++, offset++) {
                filter_flat(i) = data[offset];
            }
        }
        
        // Create TensorFlow scope and session
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input placeholders
        auto input_ph = tensorflow::ops::Placeholder(root, dtype, 
            tensorflow::ops::Placeholder::Shape(input_shape));
        auto filter_ph = tensorflow::ops::Placeholder(root, dtype,
            tensorflow::ops::Placeholder::Shape(filter_shape));
        
        // Set up strides and rates
        std::vector<int> strides = {1, static_cast<int>(stride_height), static_cast<int>(stride_width), 1};
        std::vector<int> rates = {1, static_cast<int>(rate_height), static_cast<int>(rate_width), 1};
        
        // Create Dilation2D operation
        auto dilation_op = tensorflow::ops::Dilation2D(
            root,
            input_ph,
            filter_ph,
            strides,
            rates,
            use_same_padding ? "SAME" : "VALID"
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run(
            {{input_ph, input_tensor}, {filter_ph, filter_tensor}},
            {dilation_op},
            &outputs
        );
        
        if (!status.ok()) {
            std::cout << "TensorFlow operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output tensor properties
        if (!outputs.empty()) {
            const tensorflow::Tensor& output = outputs[0];
            if (output.dtype() != dtype) {
                std::cout << "Output dtype mismatch" << std::endl;
                return 0;
            }
            
            // Check output shape dimensions
            if (output.dims() != 4) {
                std::cout << "Output should have 4 dimensions" << std::endl;
                return 0;
            }
            
            // Verify batch and depth dimensions are preserved
            if (output.dim_size(0) != batch || output.dim_size(3) != depth) {
                std::cout << "Batch or depth dimension mismatch" << std::endl;
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