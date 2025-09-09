#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/training_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/graph.pb.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions and parameters from fuzz input
        int32_t var_dim = (data[offset] % 10) + 1;
        offset++;
        int32_t num_indices = (data[offset] % var_dim) + 1;
        offset++;
        
        // Extract scalar values
        float lr_val = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        if (offset >= size) return 0;
        
        float rho_val = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        if (offset >= size) return 0;
        
        float epsilon_val = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        if (offset >= size) return 0;
        
        bool use_locking = data[offset] % 2;
        offset++;
        
        // Clamp values to reasonable ranges
        lr_val = std::abs(lr_val);
        if (lr_val > 1.0f) lr_val = 0.01f;
        if (lr_val < 1e-8f) lr_val = 1e-6f;
        
        rho_val = std::abs(rho_val);
        if (rho_val > 1.0f) rho_val = 0.95f;
        if (rho_val < 0.0f) rho_val = 0.9f;
        
        epsilon_val = std::abs(epsilon_val);
        if (epsilon_val > 1.0f) epsilon_val = 1e-6f;
        if (epsilon_val < 1e-10f) epsilon_val = 1e-8f;
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create variable tensors
        tensorflow::TensorShape var_shape({var_dim});
        tensorflow::Tensor var_tensor(tensorflow::DT_FLOAT, var_shape);
        tensorflow::Tensor accum_tensor(tensorflow::DT_FLOAT, var_shape);
        tensorflow::Tensor accum_update_tensor(tensorflow::DT_FLOAT, var_shape);
        tensorflow::Tensor grad_tensor(tensorflow::DT_FLOAT, var_shape);
        
        // Initialize tensors with fuzz data
        auto var_flat = var_tensor.flat<float>();
        auto accum_flat = accum_tensor.flat<float>();
        auto accum_update_flat = accum_update_tensor.flat<float>();
        auto grad_flat = grad_tensor.flat<float>();
        
        for (int i = 0; i < var_dim && offset + sizeof(float) <= size; i++) {
            var_flat(i) = *reinterpret_cast<const float*>(data + offset);
            offset += sizeof(float);
            if (offset >= size) break;
        }
        
        for (int i = 0; i < var_dim && offset + sizeof(float) <= size; i++) {
            accum_flat(i) = std::abs(*reinterpret_cast<const float*>(data + offset));
            offset += sizeof(float);
            if (offset >= size) break;
        }
        
        for (int i = 0; i < var_dim && offset + sizeof(float) <= size; i++) {
            accum_update_flat(i) = std::abs(*reinterpret_cast<const float*>(data + offset));
            offset += sizeof(float);
            if (offset >= size) break;
        }
        
        for (int i = 0; i < var_dim && offset + sizeof(float) <= size; i++) {
            grad_flat(i) = *reinterpret_cast<const float*>(data + offset);
            offset += sizeof(float);
            if (offset >= size) break;
        }
        
        // Create indices tensor
        tensorflow::TensorShape indices_shape({num_indices});
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, indices_shape);
        auto indices_flat = indices_tensor.flat<int32_t>();
        
        for (int i = 0; i < num_indices && offset + sizeof(int32_t) <= size; i++) {
            indices_flat(i) = std::abs(*reinterpret_cast<const int32_t*>(data + offset)) % var_dim;
            offset += sizeof(int32_t);
            if (offset >= size) break;
        }
        
        // Create scalar tensors
        tensorflow::Tensor lr_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor rho_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor epsilon_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        
        lr_tensor.scalar<float>()() = lr_val;
        rho_tensor.scalar<float>()() = rho_val;
        epsilon_tensor.scalar<float>()() = epsilon_val;
        
        // Create Variable nodes
        auto var_node = tensorflow::ops::Variable(root, var_shape, tensorflow::DT_FLOAT);
        auto accum_node = tensorflow::ops::Variable(root, var_shape, tensorflow::DT_FLOAT);
        auto accum_update_node = tensorflow::ops::Variable(root, var_shape, tensorflow::DT_FLOAT);
        
        // Create constant nodes for other inputs
        auto lr_node = tensorflow::ops::Const(root, lr_tensor);
        auto rho_node = tensorflow::ops::Const(root, rho_tensor);
        auto epsilon_node = tensorflow::ops::Const(root, epsilon_tensor);
        auto grad_node = tensorflow::ops::Const(root, grad_tensor);
        auto indices_node = tensorflow::ops::Const(root, indices_tensor);
        
        // Create SparseApplyAdadelta operation
        auto sparse_apply_adadelta = tensorflow::ops::SparseApplyAdadelta(
            root,
            var_node,
            accum_node,
            accum_update_node,
            lr_node,
            rho_node,
            epsilon_node,
            grad_node,
            indices_node,
            tensorflow::ops::SparseApplyAdadelta::UseLocking(use_locking)
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        
        // Initialize variables
        std::vector<tensorflow::Tensor> init_outputs;
        session.Run({var_node.initializer(), accum_node.initializer(), accum_update_node.initializer()}, &init_outputs);
        
        // Assign initial values
        auto assign_var = tensorflow::ops::Assign(root, var_node, tensorflow::ops::Const(root, var_tensor));
        auto assign_accum = tensorflow::ops::Assign(root, accum_node, tensorflow::ops::Const(root, accum_tensor));
        auto assign_accum_update = tensorflow::ops::Assign(root, accum_update_node, tensorflow::ops::Const(root, accum_update_tensor));
        
        std::vector<tensorflow::Tensor> assign_outputs;
        session.Run({assign_var, assign_accum, assign_accum_update}, &assign_outputs);
        
        // Run the SparseApplyAdadelta operation
        std::vector<tensorflow::Tensor> outputs;
        auto status = session.Run({sparse_apply_adadelta}, &outputs);
        
        if (!status.ok()) {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}