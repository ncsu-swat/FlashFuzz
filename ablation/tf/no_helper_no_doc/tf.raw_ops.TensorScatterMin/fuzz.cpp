#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/kernels/scatter_functor.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions and parameters from fuzz input
        uint32_t tensor_dim1 = *reinterpret_cast<const uint32_t*>(data + offset) % 10 + 1;
        offset += 4;
        uint32_t tensor_dim2 = *reinterpret_cast<const uint32_t*>(data + offset) % 10 + 1;
        offset += 4;
        uint32_t indices_count = *reinterpret_cast<const uint32_t*>(data + offset) % 5 + 1;
        offset += 4;
        uint32_t updates_count = indices_count;
        offset += 4;
        
        if (offset >= size) return 0;
        
        // Create TensorFlow scope
        auto root = tensorflow::Scope::NewRootScope();
        
        // Create input tensor
        tensorflow::TensorShape tensor_shape({static_cast<int64_t>(tensor_dim1), static_cast<int64_t>(tensor_dim2)});
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensor_shape);
        auto tensor_flat = input_tensor.flat<float>();
        
        // Fill tensor with fuzz data
        for (int i = 0; i < tensor_flat.size() && offset + 4 <= size; ++i) {
            float val = *reinterpret_cast<const float*>(data + offset);
            if (std::isfinite(val)) {
                tensor_flat(i) = val;
            } else {
                tensor_flat(i) = 1.0f;
            }
            offset += 4;
        }
        
        // Create indices tensor
        tensorflow::TensorShape indices_shape({static_cast<int64_t>(indices_count), 2});
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, indices_shape);
        auto indices_flat = indices_tensor.flat<int32_t>();
        
        for (int i = 0; i < indices_flat.size() && offset + 4 <= size; ++i) {
            int32_t idx = *reinterpret_cast<const int32_t*>(data + offset);
            if (i % 2 == 0) {
                indices_flat(i) = std::abs(idx) % tensor_dim1;
            } else {
                indices_flat(i) = std::abs(idx) % tensor_dim2;
            }
            offset += 4;
        }
        
        // Create updates tensor
        tensorflow::TensorShape updates_shape({static_cast<int64_t>(updates_count)});
        tensorflow::Tensor updates_tensor(tensorflow::DT_FLOAT, updates_shape);
        auto updates_flat = updates_tensor.flat<float>();
        
        for (int i = 0; i < updates_flat.size() && offset + 4 <= size; ++i) {
            float val = *reinterpret_cast<const float*>(data + offset);
            if (std::isfinite(val)) {
                updates_flat(i) = val;
            } else {
                updates_flat(i) = 1.0f;
            }
            offset += 4;
        }
        
        // Create placeholder ops
        auto tensor_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto indices_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto updates_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        // Create TensorScatterMin operation
        auto scatter_min_op = tensorflow::ops::TensorScatterMin(root, 
                                                               tensor_placeholder, 
                                                               indices_placeholder, 
                                                               updates_placeholder);
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run({{tensor_placeholder, input_tensor},
                                                 {indices_placeholder, indices_tensor},
                                                 {updates_placeholder, updates_tensor}},
                                                {scatter_min_op}, &outputs);
        
        if (!status.ok()) {
            // Operation failed, but this is acceptable for fuzzing
            return 0;
        }
        
        // Verify output tensor has correct shape
        if (!outputs.empty()) {
            const tensorflow::Tensor& result = outputs[0];
            if (result.shape().dims() == 2 && 
                result.shape().dim_size(0) == tensor_dim1 && 
                result.shape().dim_size(1) == tensor_dim2) {
                // Success case
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}