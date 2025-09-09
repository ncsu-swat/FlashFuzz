#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/public/session.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions for first tensor
        uint32_t dim1_x = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        uint32_t dim1_y = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        
        // Extract dimensions for second tensor
        uint32_t dim2_x = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        uint32_t dim2_y = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        
        // Limit dimensions to reasonable values
        dim1_x = (dim1_x % 10) + 1;
        dim1_y = (dim1_y % 10) + 1;
        dim2_x = (dim2_x % 10) + 1;
        dim2_y = (dim2_y % 10) + 1;
        
        size_t tensor1_size = dim1_x * dim1_y;
        size_t tensor2_size = dim2_x * dim2_y;
        
        if (offset + tensor1_size + tensor2_size > size) return 0;
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create first tensor
        tensorflow::TensorShape shape1({static_cast<int64_t>(dim1_x), static_cast<int64_t>(dim1_y)});
        tensorflow::Tensor tensor1(tensorflow::DT_BOOL, shape1);
        auto tensor1_flat = tensor1.flat<bool>();
        
        for (size_t i = 0; i < tensor1_size && offset < size; ++i) {
            tensor1_flat(i) = (data[offset] % 2) == 1;
            offset++;
        }
        
        // Create second tensor
        tensorflow::TensorShape shape2({static_cast<int64_t>(dim2_x), static_cast<int64_t>(dim2_y)});
        tensorflow::Tensor tensor2(tensorflow::DT_BOOL, shape2);
        auto tensor2_flat = tensor2.flat<bool>();
        
        for (size_t i = 0; i < tensor2_size && offset < size; ++i) {
            tensor2_flat(i) = (data[offset] % 2) == 1;
            offset++;
        }
        
        // Create placeholder operations
        auto x = tensorflow::ops::Placeholder(root, tensorflow::DT_BOOL);
        auto y = tensorflow::ops::Placeholder(root, tensorflow::DT_BOOL);
        
        // Create LogicalOr operation
        auto logical_or = tensorflow::ops::LogicalOr(root, x, y);
        
        // Create session and run the operation
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({{x, tensor1}, {y, tensor2}}, {logical_or}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Operation succeeded, verify output shape and values
            const tensorflow::Tensor& result = outputs[0];
            if (result.dtype() == tensorflow::DT_BOOL) {
                auto result_flat = result.flat<bool>();
                // Basic validation - just access the data to ensure it's valid
                for (int i = 0; i < result.NumElements(); ++i) {
                    volatile bool val = result_flat(i);
                    (void)val; // Suppress unused variable warning
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