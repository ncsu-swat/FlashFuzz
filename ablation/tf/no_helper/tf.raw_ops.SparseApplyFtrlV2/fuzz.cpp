#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/training_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/framework/graph.pb.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions and parameters from fuzz data
        int32_t var_dim0 = *reinterpret_cast<const int32_t*>(data + offset) % 100 + 1;
        offset += sizeof(int32_t);
        int32_t var_dim1 = *reinterpret_cast<const int32_t*>(data + offset) % 100 + 1;
        offset += sizeof(int32_t);
        int32_t num_indices = *reinterpret_cast<const int32_t*>(data + offset) % std::min(var_dim0, 10) + 1;
        offset += sizeof(int32_t);
        
        bool use_locking = (*reinterpret_cast<const uint8_t*>(data + offset)) % 2;
        offset += sizeof(uint8_t);
        bool multiply_linear_by_lr = (*reinterpret_cast<const uint8_t*>(data + offset)) % 2;
        offset += sizeof(uint8_t);
        
        // Extract scalar values
        float lr_val = 0.01f;
        float l1_val = 0.1f;
        float l2_val = 0.1f;
        float l2_shrinkage_val = 0.01f;
        float lr_power_val = -0.5f;
        
        if (offset + 5 * sizeof(float) <= size) {
            lr_val = std::abs(*reinterpret_cast<const float*>(data + offset)) + 0.001f;
            offset += sizeof(float);
            l1_val = std::abs(*reinterpret_cast<const float*>(data + offset));
            offset += sizeof(float);
            l2_val = std::abs(*reinterpret_cast<const float*>(data + offset));
            offset += sizeof(float);
            l2_shrinkage_val = std::abs(*reinterpret_cast<const float*>(data + offset));
            offset += sizeof(float);
            lr_power_val = *reinterpret_cast<const float*>(data + offset);
            if (lr_power_val >= 0) lr_power_val = -0.5f;
            offset += sizeof(float);
        }
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create variable tensors
        tensorflow::TensorShape var_shape({var_dim0, var_dim1});
        tensorflow::Tensor var_tensor(tensorflow::DT_FLOAT, var_shape);
        tensorflow::Tensor accum_tensor(tensorflow::DT_FLOAT, var_shape);
        tensorflow::Tensor linear_tensor(tensorflow::DT_FLOAT, var_shape);
        
        // Initialize with small random values from fuzz data
        auto var_flat = var_tensor.flat<float>();
        auto accum_flat = accum_tensor.flat<float>();
        auto linear_flat = linear_tensor.flat<float>();
        
        for (int i = 0; i < var_flat.size() && offset < size; ++i) {
            var_flat(i) = (offset < size) ? 
                (*reinterpret_cast<const float*>(data + (offset % (size - sizeof(float))))) * 0.01f : 0.1f;
            accum_flat(i) = 0.1f; // Keep positive for numerical stability
            linear_flat(i) = (offset + 4 < size) ? 
                (*reinterpret_cast<const float*>(data + ((offset + 4) % (size - sizeof(float))))) * 0.01f : 0.0f;
            offset = (offset + 8) % size;
        }
        
        // Create grad tensor
        tensorflow::TensorShape grad_shape({num_indices, var_dim1});
        tensorflow::Tensor grad_tensor(tensorflow::DT_FLOAT, grad_shape);
        auto grad_flat = grad_tensor.flat<float>();
        for (int i = 0; i < grad_flat.size() && offset < size; ++i) {
            grad_flat(i) = (offset < size) ? 
                (*reinterpret_cast<const float*>(data + (offset % (size - sizeof(float))))) * 0.01f : 0.01f;
            offset = (offset + 4) % size;
        }
        
        // Create indices tensor
        tensorflow::TensorShape indices_shape({num_indices});
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, indices_shape);
        auto indices_flat = indices_tensor.flat<int32_t>();
        for (int i = 0; i < num_indices; ++i) {
            indices_flat(i) = i % var_dim0;
        }
        
        // Create scalar tensors
        tensorflow::Tensor lr_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        lr_tensor.scalar<float>()() = lr_val;
        
        tensorflow::Tensor l1_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        l1_tensor.scalar<float>()() = l1_val;
        
        tensorflow::Tensor l2_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        l2_tensor.scalar<float>()() = l2_val;
        
        tensorflow::Tensor l2_shrinkage_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        l2_shrinkage_tensor.scalar<float>()() = l2_shrinkage_val;
        
        tensorflow::Tensor lr_power_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        lr_power_tensor.scalar<float>()() = lr_power_val;
        
        // Create input nodes
        auto var_node = tensorflow::ops::Const(root, var_tensor);
        auto accum_node = tensorflow::ops::Const(root, accum_tensor);
        auto linear_node = tensorflow::ops::Const(root, linear_tensor);
        auto grad_node = tensorflow::ops::Const(root, grad_tensor);
        auto indices_node = tensorflow::ops::Const(root, indices_tensor);
        auto lr_node = tensorflow::ops::Const(root, lr_tensor);
        auto l1_node = tensorflow::ops::Const(root, l1_tensor);
        auto l2_node = tensorflow::ops::Const(root, l2_tensor);
        auto l2_shrinkage_node = tensorflow::ops::Const(root, l2_shrinkage_tensor);
        auto lr_power_node = tensorflow::ops::Const(root, lr_power_tensor);
        
        // Create SparseApplyFtrlV2 operation
        auto sparse_apply_ftrl = tensorflow::ops::SparseApplyFtrlV2(
            root,
            var_node,
            accum_node,
            linear_node,
            grad_node,
            indices_node,
            lr_node,
            l1_node,
            l2_node,
            l2_shrinkage_node,
            lr_power_node,
            tensorflow::ops::SparseApplyFtrlV2::UseLocking(use_locking)
                .MultiplyLinearByLr(multiply_linear_by_lr)
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        auto status = session.Run({sparse_apply_ftrl}, &outputs);
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