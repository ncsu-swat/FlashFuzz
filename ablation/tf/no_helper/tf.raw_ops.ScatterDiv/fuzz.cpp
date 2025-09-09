#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/ops/state_ops.h>
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
        uint32_t num_indices = *reinterpret_cast<const uint32_t*>(data + offset) % ref_dim0 + 1;
        offset += 4;
        bool use_locking = (*reinterpret_cast<const uint32_t*>(data + offset)) % 2;
        offset += 4;
        
        if (offset + num_indices * 4 + ref_dim0 * ref_dim1 * 4 + num_indices * ref_dim1 * 4 > size) {
            return 0;
        }
        
        // Create TensorFlow scope and session
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create ref tensor (variable)
        tensorflow::TensorShape ref_shape({static_cast<int64_t>(ref_dim0), static_cast<int64_t>(ref_dim1)});
        tensorflow::Tensor ref_tensor(tensorflow::DT_FLOAT, ref_shape);
        auto ref_flat = ref_tensor.flat<float>();
        
        // Fill ref tensor with data from fuzz input
        for (int i = 0; i < ref_dim0 * ref_dim1 && offset + 4 <= size; ++i) {
            float val = *reinterpret_cast<const float*>(data + offset);
            // Avoid division by zero by ensuring non-zero values
            if (val == 0.0f) val = 1.0f;
            ref_flat(i) = val;
            offset += 4;
        }
        
        // Create indices tensor
        tensorflow::TensorShape indices_shape({static_cast<int64_t>(num_indices)});
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, indices_shape);
        auto indices_flat = indices_tensor.flat<int32_t>();
        
        for (int i = 0; i < num_indices && offset + 4 <= size; ++i) {
            int32_t idx = *reinterpret_cast<const int32_t*>(data + offset) % ref_dim0;
            if (idx < 0) idx = -idx;
            indices_flat(i) = idx;
            offset += 4;
        }
        
        // Create updates tensor
        tensorflow::TensorShape updates_shape({static_cast<int64_t>(num_indices), static_cast<int64_t>(ref_dim1)});
        tensorflow::Tensor updates_tensor(tensorflow::DT_FLOAT, updates_shape);
        auto updates_flat = updates_tensor.flat<float>();
        
        for (int i = 0; i < num_indices * ref_dim1 && offset + 4 <= size; ++i) {
            float val = *reinterpret_cast<const float*>(data + offset);
            // Avoid division by zero
            if (val == 0.0f) val = 1.0f;
            updates_flat(i) = val;
            offset += 4;
        }
        
        // Create variable node
        auto var = tensorflow::ops::Variable(root, ref_shape, tensorflow::DT_FLOAT);
        auto assign = tensorflow::ops::Assign(root, var, tensorflow::ops::Const(root, ref_tensor));
        
        // Create ScatterDiv operation
        auto scatter_div = tensorflow::ops::ScatterDiv(
            root,
            var,
            tensorflow::ops::Const(root, indices_tensor),
            tensorflow::ops::Const(root, updates_tensor),
            tensorflow::ops::ScatterDiv::UseLocking(use_locking)
        );
        
        // Create session and run operations
        tensorflow::ClientSession session(root);
        
        // Initialize variable
        std::vector<tensorflow::Tensor> assign_outputs;
        auto assign_status = session.Run({assign}, &assign_outputs);
        if (!assign_status.ok()) {
            return 0;
        }
        
        // Run ScatterDiv
        std::vector<tensorflow::Tensor> outputs;
        auto status = session.Run({scatter_div}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Verify output tensor properties
            const auto& output = outputs[0];
            if (output.dtype() == tensorflow::DT_FLOAT && 
                output.shape().dims() == 2 &&
                output.shape().dim_size(0) == ref_dim0 &&
                output.shape().dim_size(1) == ref_dim1) {
                // Basic validation passed
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}