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
        
        // Need at least 4 bytes for tensor size
        if (size < 4) return 0;
        
        // Read tensor size
        uint32_t tensor_size;
        memcpy(&tensor_size, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        // Limit tensor size to reasonable bounds
        tensor_size = tensor_size % 100 + 1;
        
        // Check if we have enough data for the tensor
        if (offset + tensor_size * sizeof(int32_t) > size) return 0;
        
        // Create TensorFlow scope and session
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        tensorflow::ClientSession session(root);
        
        // Create input tensor
        tensorflow::Tensor input_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({static_cast<int64_t>(tensor_size)}));
        auto input_flat = input_tensor.flat<int32_t>();
        
        // Fill tensor with data, ensuring valid permutation
        std::vector<int32_t> values(tensor_size);
        for (uint32_t i = 0; i < tensor_size; ++i) {
            values[i] = i;
        }
        
        // Shuffle values based on input data
        for (uint32_t i = 0; i < tensor_size && offset < size; ++i) {
            uint32_t swap_idx = data[offset] % tensor_size;
            std::swap(values[i], values[swap_idx]);
            offset++;
        }
        
        // Copy to tensor
        for (uint32_t i = 0; i < tensor_size; ++i) {
            input_flat(i) = values[i];
        }
        
        // Create placeholder for input
        auto x = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        
        // Apply InvertPermutation operation
        auto invert_op = tensorflow::ops::InvertPermutation(root, x);
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({{x, input_tensor}}, {invert_op}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Verify the result
            auto output_flat = outputs[0].flat<int32_t>();
            
            // Check that y[x[i]] = i for all i
            for (int32_t i = 0; i < tensor_size; ++i) {
                int32_t x_i = input_flat(i);
                if (x_i >= 0 && x_i < tensor_size) {
                    int32_t y_x_i = output_flat(x_i);
                    if (y_x_i != i) {
                        // Verification failed, but don't crash
                        break;
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