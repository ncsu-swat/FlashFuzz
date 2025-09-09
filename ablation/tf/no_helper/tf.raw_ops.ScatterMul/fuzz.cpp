#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/public/session.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions and parameters from fuzz input
        uint32_t ref_dim0 = *reinterpret_cast<const uint32_t*>(data + offset) % 10 + 1;
        offset += 4;
        uint32_t ref_dim1 = *reinterpret_cast<const uint32_t*>(data + offset) % 10 + 1;
        offset += 4;
        uint32_t num_indices = *reinterpret_cast<const uint32_t*>(data + offset) % 5 + 1;
        offset += 4;
        bool use_locking = (*reinterpret_cast<const uint32_t*>(data + offset)) % 2;
        offset += 4;
        
        if (offset + ref_dim0 * ref_dim1 * sizeof(float) + num_indices * sizeof(int32_t) + num_indices * ref_dim1 * sizeof(float) > size) {
            return 0;
        }
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create ref tensor (mutable)
        tensorflow::TensorShape ref_shape({static_cast<int64_t>(ref_dim0), static_cast<int64_t>(ref_dim1)});
        tensorflow::Tensor ref_tensor(tensorflow::DT_FLOAT, ref_shape);
        auto ref_flat = ref_tensor.flat<float>();
        
        for (int i = 0; i < ref_dim0 * ref_dim1; ++i) {
            if (offset + sizeof(float) > size) break;
            float val = *reinterpret_cast<const float*>(data + offset);
            // Clamp to reasonable range to avoid overflow
            val = std::max(-100.0f, std::min(100.0f, val));
            if (std::isnan(val) || std::isinf(val)) val = 1.0f;
            ref_flat(i) = val;
            offset += sizeof(float);
        }
        
        // Create indices tensor
        tensorflow::TensorShape indices_shape({static_cast<int64_t>(num_indices)});
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, indices_shape);
        auto indices_flat = indices_tensor.flat<int32_t>();
        
        for (int i = 0; i < num_indices; ++i) {
            if (offset + sizeof(int32_t) > size) break;
            int32_t idx = *reinterpret_cast<const int32_t*>(data + offset);
            // Ensure valid index range
            idx = std::abs(idx) % ref_dim0;
            indices_flat(i) = idx;
            offset += sizeof(int32_t);
        }
        
        // Create updates tensor
        tensorflow::TensorShape updates_shape({static_cast<int64_t>(num_indices), static_cast<int64_t>(ref_dim1)});
        tensorflow::Tensor updates_tensor(tensorflow::DT_FLOAT, updates_shape);
        auto updates_flat = updates_tensor.flat<float>();
        
        for (int i = 0; i < num_indices * ref_dim1; ++i) {
            if (offset + sizeof(float) > size) break;
            float val = *reinterpret_cast<const float*>(data + offset);
            // Clamp to reasonable range to avoid overflow
            val = std::max(-10.0f, std::min(10.0f, val));
            if (std::isnan(val) || std::isinf(val)) val = 1.0f;
            updates_flat(i) = val;
            offset += sizeof(float);
        }
        
        // Create Variable node for ref
        auto ref_var = tensorflow::ops::Variable(root, ref_shape, tensorflow::DT_FLOAT);
        auto assign_ref = tensorflow::ops::Assign(root, ref_var, tensorflow::ops::Const(root, ref_tensor));
        
        // Create constant nodes for indices and updates
        auto indices_const = tensorflow::ops::Const(root, indices_tensor);
        auto updates_const = tensorflow::ops::Const(root, updates_tensor);
        
        // Create ScatterMul operation
        auto scatter_mul = tensorflow::ops::ScatterMul(root, ref_var, indices_const, updates_const,
                                                      tensorflow::ops::ScatterMul::UseLocking(use_locking));
        
        // Create session and run
        tensorflow::ClientSession session(root);
        
        // Initialize variable
        std::vector<tensorflow::Tensor> init_outputs;
        tensorflow::Status init_status = session.Run({assign_ref}, &init_outputs);
        if (!init_status.ok()) {
            return 0;
        }
        
        // Run ScatterMul
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({scatter_mul}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Verify output tensor properties
            const tensorflow::Tensor& result = outputs[0];
            if (result.dtype() == tensorflow::DT_FLOAT && 
                result.shape().dims() == 2 &&
                result.dim_size(0) == ref_dim0 &&
                result.dim_size(1) == ref_dim1) {
                
                // Basic validation of result values
                auto result_flat = result.flat<float>();
                for (int i = 0; i < result.NumElements(); ++i) {
                    float val = result_flat(i);
                    if (std::isnan(val) || std::isinf(val)) {
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