#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/resource_mgr.h>
#include <tensorflow/core/framework/resource_var.h>
#include <tensorflow/core/kernels/training_ops.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/resource_variable_ops.h>
#include <tensorflow/cc/ops/training_ops.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions and parameters from fuzz input
        int32_t var_dim0 = *reinterpret_cast<const int32_t*>(data + offset) % 100 + 1;
        offset += sizeof(int32_t);
        int32_t var_dim1 = *reinterpret_cast<const int32_t*>(data + offset) % 100 + 1;
        offset += sizeof(int32_t);
        
        float lr_val = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        float l1_val = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        float l2_val = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        
        int32_t num_indices = *reinterpret_cast<const int32_t*>(data + offset) % 10 + 1;
        offset += sizeof(int32_t);
        
        bool use_locking = (*reinterpret_cast<const uint8_t*>(data + offset)) % 2;
        offset += sizeof(uint8_t);
        
        if (offset + num_indices * sizeof(int32_t) + num_indices * var_dim1 * sizeof(float) > size) {
            return 0;
        }
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create variable tensors
        tensorflow::TensorShape var_shape({var_dim0, var_dim1});
        tensorflow::Tensor var_init(tensorflow::DT_FLOAT, var_shape);
        auto var_flat = var_init.flat<float>();
        for (int i = 0; i < var_flat.size(); ++i) {
            var_flat(i) = 0.1f;
        }
        
        tensorflow::Tensor accum_init(tensorflow::DT_FLOAT, var_shape);
        auto accum_flat = accum_init.flat<float>();
        for (int i = 0; i < accum_flat.size(); ++i) {
            accum_flat(i) = 0.1f;
        }
        
        // Create resource variables
        auto var = tensorflow::ops::VarHandleOp(root, tensorflow::DT_FLOAT, var_shape);
        auto accum = tensorflow::ops::VarHandleOp(root, tensorflow::DT_FLOAT, var_shape);
        
        // Initialize variables
        auto var_assign = tensorflow::ops::AssignVariableOp(root, var, var_init);
        auto accum_assign = tensorflow::ops::AssignVariableOp(root, accum, accum_init);
        
        // Create scalar tensors
        auto lr = tensorflow::ops::Const(root, lr_val);
        auto l1 = tensorflow::ops::Const(root, l1_val);
        auto l2 = tensorflow::ops::Const(root, l2_val);
        
        // Create indices tensor
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({num_indices}));
        auto indices_flat = indices_tensor.flat<int32_t>();
        for (int i = 0; i < num_indices; ++i) {
            indices_flat(i) = (*reinterpret_cast<const int32_t*>(data + offset)) % var_dim0;
            offset += sizeof(int32_t);
        }
        auto indices = tensorflow::ops::Const(root, indices_tensor);
        
        // Create gradient tensor
        tensorflow::Tensor grad_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({num_indices, var_dim1}));
        auto grad_flat = grad_tensor.flat<float>();
        for (int i = 0; i < grad_flat.size() && offset + sizeof(float) <= size; ++i) {
            grad_flat(i) = *reinterpret_cast<const float*>(data + offset);
            offset += sizeof(float);
        }
        auto grad = tensorflow::ops::Const(root, grad_tensor);
        
        // Create the ResourceSparseApplyProximalAdagrad operation
        auto apply_op = tensorflow::ops::ResourceSparseApplyProximalAdagrad(
            root, var, accum, lr, l1, l2, grad, indices,
            tensorflow::ops::ResourceSparseApplyProximalAdagrad::UseLocking(use_locking));
        
        // Create session and run
        tensorflow::ClientSession session(root);
        
        // Initialize variables first
        TF_CHECK_OK(session.Run({var_assign, accum_assign}, nullptr));
        
        // Run the operation
        TF_CHECK_OK(session.Run({apply_op}, nullptr));
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}