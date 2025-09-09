#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/version.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/resource_mgr.h>
#include <tensorflow/core/framework/resource_var.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/resource_variable_ops.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions and parameters from fuzz data
        int32_t var_dim = *reinterpret_cast<const int32_t*>(data + offset) % 100 + 1;
        offset += sizeof(int32_t);
        
        int32_t indices_size = *reinterpret_cast<const int32_t*>(data + offset) % 10 + 1;
        offset += sizeof(int32_t);
        
        float lr_val = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        
        float l1_val = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        
        float l2_val = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        
        int64_t global_step_val = *reinterpret_cast<const int64_t*>(data + offset);
        offset += sizeof(int64_t);
        
        bool use_locking = (data[offset] % 2) == 1;
        offset += 1;
        
        if (offset >= size) return 0;
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create variable tensors
        tensorflow::TensorShape var_shape({var_dim});
        auto var_init = tensorflow::ops::Const(root, tensorflow::Tensor(tensorflow::DT_FLOAT, var_shape));
        auto var = tensorflow::ops::VarHandleOp(root, tensorflow::DT_FLOAT, var_shape);
        
        auto grad_accum_init = tensorflow::ops::Const(root, tensorflow::Tensor(tensorflow::DT_FLOAT, var_shape));
        auto grad_accum = tensorflow::ops::VarHandleOp(root, tensorflow::DT_FLOAT, var_shape);
        
        auto grad_sq_accum_init = tensorflow::ops::Const(root, tensorflow::Tensor(tensorflow::DT_FLOAT, var_shape));
        auto grad_sq_accum = tensorflow::ops::VarHandleOp(root, tensorflow::DT_FLOAT, var_shape);
        
        // Create gradient tensor
        tensorflow::TensorShape grad_shape({indices_size});
        tensorflow::Tensor grad_tensor(tensorflow::DT_FLOAT, grad_shape);
        auto grad_flat = grad_tensor.flat<float>();
        for (int i = 0; i < indices_size && offset + sizeof(float) <= size; ++i) {
            grad_flat(i) = *reinterpret_cast<const float*>(data + offset);
            offset += sizeof(float);
        }
        auto grad = tensorflow::ops::Const(root, grad_tensor);
        
        // Create indices tensor
        tensorflow::TensorShape indices_shape({indices_size});
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, indices_shape);
        auto indices_flat = indices_tensor.flat<int32_t>();
        for (int i = 0; i < indices_size && offset + sizeof(int32_t) <= size; ++i) {
            indices_flat(i) = (*reinterpret_cast<const int32_t*>(data + offset)) % var_dim;
            offset += sizeof(int32_t);
        }
        auto indices = tensorflow::ops::Const(root, indices_tensor);
        
        // Create scalar tensors
        auto lr = tensorflow::ops::Const(root, lr_val);
        auto l1 = tensorflow::ops::Const(root, l1_val);
        auto l2 = tensorflow::ops::Const(root, l2_val);
        auto global_step = tensorflow::ops::Const(root, global_step_val);
        
        // Initialize variables
        tensorflow::ClientSession session(root);
        
        TF_CHECK_OK(session.Run({tensorflow::ops::AssignVariableOp(root, var, var_init)}, nullptr));
        TF_CHECK_OK(session.Run({tensorflow::ops::AssignVariableOp(root, grad_accum, grad_accum_init)}, nullptr));
        TF_CHECK_OK(session.Run({tensorflow::ops::AssignVariableOp(root, grad_sq_accum, grad_sq_accum_init)}, nullptr));
        
        // Create the ResourceSparseApplyAdagradDA operation
        auto apply_op = tensorflow::ops::ResourceSparseApplyAdagradDA(
            root,
            var,
            grad_accum,
            grad_sq_accum,
            grad,
            indices,
            lr,
            l1,
            l2,
            global_step,
            tensorflow::ops::ResourceSparseApplyAdagradDA::UseLocking(use_locking)
        );
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({apply_op}, &outputs);
        
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