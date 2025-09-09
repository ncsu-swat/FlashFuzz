#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/resource_var.h>
#include <tensorflow/core/framework/resource_mgr.h>
#include <tensorflow/core/kernels/resource_variable_ops.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/status.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/kernels/sparse_apply_momentum_op.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/resource_variable_ops.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract parameters from fuzzer input
        int32_t var_dim = *reinterpret_cast<const int32_t*>(data + offset) % 100 + 1;
        offset += sizeof(int32_t);
        
        int32_t indices_size = *reinterpret_cast<const int32_t*>(data + offset) % 10 + 1;
        offset += sizeof(int32_t);
        
        float lr_val = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        
        float momentum_val = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        
        bool use_locking = (*reinterpret_cast<const uint8_t*>(data + offset)) % 2;
        offset += sizeof(uint8_t);
        
        bool use_nesterov = (*reinterpret_cast<const uint8_t*>(data + offset)) % 2;
        offset += sizeof(uint8_t);
        
        // Clamp values to reasonable ranges
        lr_val = std::max(-10.0f, std::min(10.0f, lr_val));
        momentum_val = std::max(-1.0f, std::min(1.0f, momentum_val));
        var_dim = std::max(1, std::min(100, var_dim));
        indices_size = std::max(1, std::min(var_dim, indices_size));
        
        // Create TensorFlow session
        tensorflow::SessionOptions session_options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(session_options));
        
        // Create scope
        auto root = tensorflow::Scope::NewRootScope();
        
        // Create variable tensors
        tensorflow::TensorShape var_shape({var_dim});
        auto var_init = tensorflow::ops::Const(root, tensorflow::Tensor(tensorflow::DT_FLOAT, var_shape));
        auto var = tensorflow::ops::VarHandleOp(root, tensorflow::DT_FLOAT, var_shape);
        auto var_assign = tensorflow::ops::AssignVariableOp(root, var, var_init);
        
        auto accum_init = tensorflow::ops::Const(root, tensorflow::Tensor(tensorflow::DT_FLOAT, var_shape));
        auto accum = tensorflow::ops::VarHandleOp(root, tensorflow::DT_FLOAT, var_shape);
        auto accum_assign = tensorflow::ops::AssignVariableOp(root, accum, accum_init);
        
        // Create learning rate tensor
        auto lr = tensorflow::ops::Const(root, lr_val);
        
        // Create momentum tensor
        auto momentum = tensorflow::ops::Const(root, momentum_val);
        
        // Create gradient tensor
        tensorflow::TensorShape grad_shape({indices_size});
        tensorflow::Tensor grad_tensor(tensorflow::DT_FLOAT, grad_shape);
        auto grad_flat = grad_tensor.flat<float>();
        for (int i = 0; i < indices_size && offset + sizeof(float) <= size; ++i) {
            float grad_val = *reinterpret_cast<const float*>(data + offset);
            grad_val = std::max(-100.0f, std::min(100.0f, grad_val));
            grad_flat(i) = grad_val;
            offset += sizeof(float);
        }
        auto grad = tensorflow::ops::Const(root, grad_tensor);
        
        // Create indices tensor
        tensorflow::TensorShape indices_shape({indices_size});
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, indices_shape);
        auto indices_flat = indices_tensor.flat<int32_t>();
        for (int i = 0; i < indices_size && offset + sizeof(int32_t) <= size; ++i) {
            int32_t idx = *reinterpret_cast<const int32_t*>(data + offset) % var_dim;
            indices_flat(i) = std::max(0, std::min(var_dim - 1, idx));
            offset += sizeof(int32_t);
        }
        auto indices = tensorflow::ops::Const(root, indices_tensor);
        
        // Create the ResourceSparseApplyMomentum operation
        auto momentum_op = tensorflow::ops::ResourceSparseApplyMomentum(
            root, var, accum, lr, grad, indices, momentum,
            tensorflow::ops::ResourceSparseApplyMomentum::UseLocking(use_locking)
                .UseNesterov(use_nesterov));
        
        // Build the graph
        tensorflow::GraphDef graph_def;
        TF_CHECK_OK(root.ToGraphDef(&graph_def));
        
        // Create and run the session
        TF_CHECK_OK(session->Create(graph_def));
        
        // Initialize variables
        std::vector<tensorflow::Tensor> init_outputs;
        TF_CHECK_OK(session->Run({}, {}, {var_assign.operation.name(), accum_assign.operation.name()}, &init_outputs));
        
        // Run the momentum operation
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session->Run({}, {}, {momentum_op.operation.name()}, &outputs);
        
        if (!status.ok()) {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
        }
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}