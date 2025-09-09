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
        
        if (size < 16) return 0;
        
        // Extract shape dimensions
        int32_t shape_dim = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        shape_dim = std::abs(shape_dim) % 4 + 1; // Limit to reasonable size
        
        // Extract alpha dimensions
        int32_t alpha_dim1 = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        alpha_dim1 = std::abs(alpha_dim1) % 4 + 1;
        
        int32_t alpha_dim2 = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        alpha_dim2 = std::abs(alpha_dim2) % 4 + 1;
        
        // Extract seeds
        int32_t seed = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        if (offset >= size) return 0;
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create shape tensor
        tensorflow::Tensor shape_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({shape_dim}));
        auto shape_flat = shape_tensor.flat<int32_t>();
        for (int i = 0; i < shape_dim; ++i) {
            shape_flat(i) = (i + 1) * 2; // Use small positive values
        }
        
        // Create alpha tensor with positive values
        tensorflow::Tensor alpha_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({alpha_dim1, alpha_dim2}));
        auto alpha_flat = alpha_tensor.flat<float>();
        for (int i = 0; i < alpha_dim1 * alpha_dim2; ++i) {
            if (offset + sizeof(float) <= size) {
                float val = *reinterpret_cast<const float*>(data + offset);
                offset += sizeof(float);
                // Ensure alpha is positive for gamma distribution
                alpha_flat(i) = std::abs(val) + 0.1f;
            } else {
                alpha_flat(i) = 1.0f; // Default positive value
            }
        }
        
        // Create input operations
        auto shape_op = tensorflow::ops::Const(root, shape_tensor);
        auto alpha_op = tensorflow::ops::Const(root, alpha_tensor);
        
        // Create RandomGamma operation
        auto random_gamma = tensorflow::ops::RandomGamma(root, shape_op, alpha_op,
            tensorflow::ops::RandomGamma::Seed(seed).Seed2(seed + 1));
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({random_gamma}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Verify output tensor properties
            const auto& output = outputs[0];
            if (output.dtype() == tensorflow::DT_FLOAT) {
                auto output_flat = output.flat<float>();
                // Check that all values are non-negative (gamma distribution property)
                for (int i = 0; i < output_flat.size(); ++i) {
                    if (output_flat(i) < 0) {
                        std::cout << "Negative value in gamma distribution output" << std::endl;
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