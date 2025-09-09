#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/random_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 20) return 0; // Need at least 20 bytes for basic inputs
        
        // Extract shape dimensions (1-3 dimensions)
        int32_t num_dims = (data[offset] % 3) + 1;
        offset++;
        
        std::vector<int32_t> shape_data;
        for (int i = 0; i < num_dims && offset < size; i++) {
            int32_t dim = (data[offset] % 10) + 1; // Dimensions 1-10
            shape_data.push_back(dim);
            offset++;
        }
        
        if (offset + 8 >= size) return 0;
        
        // Extract minval and maxval
        int32_t minval_data = *reinterpret_cast<const int32_t*>(data + offset);
        offset += 4;
        int32_t maxval_data = *reinterpret_cast<const int32_t*>(data + offset);
        offset += 4;
        
        // Ensure maxval > minval
        if (maxval_data <= minval_data) {
            maxval_data = minval_data + 1;
        }
        
        // Extract seeds
        int seed = 0;
        int seed2 = 0;
        if (offset + 8 <= size) {
            seed = *reinterpret_cast<const int32_t*>(data + offset);
            offset += 4;
            seed2 = *reinterpret_cast<const int32_t*>(data + offset);
            offset += 4;
        }
        
        // Create TensorFlow scope and session
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create shape tensor
        tensorflow::Tensor shape_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({static_cast<int64_t>(shape_data.size())}));
        auto shape_flat = shape_tensor.flat<int32_t>();
        for (size_t i = 0; i < shape_data.size(); i++) {
            shape_flat(i) = shape_data[i];
        }
        
        // Create minval tensor
        tensorflow::Tensor minval_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        minval_tensor.scalar<int32_t>()() = minval_data;
        
        // Create maxval tensor
        tensorflow::Tensor maxval_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        maxval_tensor.scalar<int32_t>()() = maxval_data;
        
        // Create input placeholders
        auto shape_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto minval_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto maxval_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        
        // Create RandomUniformInt operation
        auto random_uniform_int = tensorflow::ops::RandomUniformInt(
            root, 
            shape_placeholder, 
            minval_placeholder, 
            maxval_placeholder,
            tensorflow::ops::RandomUniformInt::Seed(seed).Seed2(seed2)
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run(
            {{shape_placeholder, shape_tensor}, 
             {minval_placeholder, minval_tensor}, 
             {maxval_placeholder, maxval_tensor}},
            {random_uniform_int}, 
            &outputs
        );
        
        if (!status.ok()) {
            std::cout << "TensorFlow operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output
        if (!outputs.empty()) {
            const tensorflow::Tensor& output = outputs[0];
            if (output.dtype() == tensorflow::DT_INT32) {
                auto output_flat = output.flat<int32_t>();
                // Basic validation - check if values are in expected range
                for (int i = 0; i < std::min(10, static_cast<int>(output_flat.size())); i++) {
                    int32_t val = output_flat(i);
                    if (val < minval_data || val >= maxval_data) {
                        std::cout << "Value out of range: " << val << " not in [" << minval_data << ", " << maxval_data << ")" << std::endl;
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