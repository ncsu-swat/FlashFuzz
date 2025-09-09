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
        int32_t var_dim = (data[offset] % 10) + 1;
        offset++;
        int32_t indices_size = (data[offset] % var_dim) + 1;
        offset++;
        
        // Extract scalar values
        float lr_val = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        if (offset >= size) return 0;
        
        float l1_val = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        if (offset >= size) return 0;
        
        float l2_val = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        if (offset >= size) return 0;
        
        int64_t global_step_val = *reinterpret_cast<const int64_t*>(data + offset);
        offset += sizeof(int64_t);
        if (offset >= size) return 0;
        
        bool use_locking = (data[offset] % 2) == 1;
        offset++;
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create variable tensors
        tensorflow::TensorShape var_shape({var_dim});
        tensorflow::Tensor var_tensor(tensorflow::DT_FLOAT, var_shape);
        tensorflow::Tensor grad_accum_tensor(tensorflow::DT_FLOAT, var_shape);
        tensorflow::Tensor grad_squared_accum_tensor(tensorflow::DT_FLOAT, var_shape);
        
        // Initialize with fuzz data
        auto var_flat = var_tensor.flat<float>();
        auto grad_accum_flat = grad_accum_tensor.flat<float>();
        auto grad_squared_accum_flat = grad_squared_accum_tensor.flat<float>();
        
        for (int i = 0; i < var_dim && offset + sizeof(float) <= size; i++) {
            var_flat(i) = *reinterpret_cast<const float*>(data + offset);
            offset += sizeof(float);
            if (offset + sizeof(float) <= size) {
                grad_accum_flat(i) = *reinterpret_cast<const float*>(data + offset);
                offset += sizeof(float);
            }
            if (offset + sizeof(float) <= size) {
                grad_squared_accum_flat(i) = *reinterpret_cast<const float*>(data + offset);
                offset += sizeof(float);
            }
        }
        
        // Create gradient tensor
        tensorflow::TensorShape grad_shape({indices_size});
        tensorflow::Tensor grad_tensor(tensorflow::DT_FLOAT, grad_shape);
        auto grad_flat = grad_tensor.flat<float>();
        
        for (int i = 0; i < indices_size && offset + sizeof(float) <= size; i++) {
            grad_flat(i) = *reinterpret_cast<const float*>(data + offset);
            offset += sizeof(float);
        }
        
        // Create indices tensor
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({indices_size}));
        auto indices_flat = indices_tensor.flat<int32_t>();
        
        for (int i = 0; i < indices_size; i++) {
            indices_flat(i) = i % var_dim; // Ensure valid indices
        }
        
        // Create scalar tensors
        tensorflow::Tensor lr_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        lr_tensor.scalar<float>()() = lr_val;
        
        tensorflow::Tensor l1_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        l1_tensor.scalar<float>()() = l1_val;
        
        tensorflow::Tensor l2_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        l2_tensor.scalar<float>()() = l2_val;
        
        tensorflow::Tensor global_step_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        global_step_tensor.scalar<int64_t>()() = global_step_val;
        
        // Create Variable ops
        auto var_op = tensorflow::ops::Variable(root, var_shape, tensorflow::DT_FLOAT);
        auto grad_accum_op = tensorflow::ops::Variable(root, var_shape, tensorflow::DT_FLOAT);
        auto grad_squared_accum_op = tensorflow::ops::Variable(root, var_shape, tensorflow::DT_FLOAT);
        
        // Create Const ops for other inputs
        auto grad_op = tensorflow::ops::Const(root, grad_tensor);
        auto indices_op = tensorflow::ops::Const(root, indices_tensor);
        auto lr_op = tensorflow::ops::Const(root, lr_tensor);
        auto l1_op = tensorflow::ops::Const(root, l1_tensor);
        auto l2_op = tensorflow::ops::Const(root, l2_tensor);
        auto global_step_op = tensorflow::ops::Const(root, global_step_tensor);
        
        // Create SparseApplyAdagradDA operation
        auto sparse_apply_adagrad_da = tensorflow::ops::SparseApplyAdagradDA(
            root,
            var_op,
            grad_accum_op,
            grad_squared_accum_op,
            grad_op,
            indices_op,
            lr_op,
            l1_op,
            l2_op,
            global_step_op,
            tensorflow::ops::SparseApplyAdagradDA::UseLocking(use_locking)
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        
        // Initialize variables
        std::vector<tensorflow::Tensor> init_outputs;
        session.Run({var_op.initializer(), grad_accum_op.initializer(), grad_squared_accum_op.initializer()}, &init_outputs);
        
        // Assign initial values
        auto assign_var = tensorflow::ops::Assign(root, var_op, tensorflow::ops::Const(root, var_tensor));
        auto assign_grad_accum = tensorflow::ops::Assign(root, grad_accum_op, tensorflow::ops::Const(root, grad_accum_tensor));
        auto assign_grad_squared = tensorflow::ops::Assign(root, grad_squared_accum_op, tensorflow::ops::Const(root, grad_squared_accum_tensor));
        
        std::vector<tensorflow::Tensor> assign_outputs;
        session.Run({assign_var, assign_grad_accum, assign_grad_squared}, &assign_outputs);
        
        // Run the SparseApplyAdagradDA operation
        std::vector<tensorflow::Tensor> outputs;
        auto status = session.Run({sparse_apply_adagrad_da}, &outputs);
        
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