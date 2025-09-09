#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/resource_mgr.h>
#include <tensorflow/core/framework/resource_var.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/framework/fake_input.h>
#include <tensorflow/core/kernels/ops_testutil.h>
#include <tensorflow/core/common_runtime/kernel_benchmark_testlib.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions and parameters from fuzz input
        int32_t var_dim0 = *reinterpret_cast<const int32_t*>(data + offset) % 100 + 1;
        offset += 4;
        int32_t var_dim1 = *reinterpret_cast<const int32_t*>(data + offset) % 100 + 1;
        offset += 4;
        int32_t indices_size = *reinterpret_cast<const int32_t*>(data + offset) % 10 + 1;
        offset += 4;
        
        float lr = *reinterpret_cast<const float*>(data + offset);
        offset += 4;
        float momentum = *reinterpret_cast<const float*>(data + offset);
        offset += 4;
        
        bool use_locking = (*reinterpret_cast<const uint8_t*>(data + offset)) % 2;
        offset += 1;
        bool use_nesterov = (*reinterpret_cast<const uint8_t*>(data + offset)) % 2;
        offset += 1;
        
        // Ensure we have enough data for tensors
        size_t required_size = offset + (var_dim0 * var_dim1 + indices_size + indices_size * var_dim1) * 4;
        if (size < required_size) return 0;
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create variable tensors
        tensorflow::Tensor var_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({var_dim0, var_dim1}));
        tensorflow::Tensor accum_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({var_dim0, var_dim1}));
        
        auto var_flat = var_tensor.flat<float>();
        auto accum_flat = accum_tensor.flat<float>();
        
        // Fill variable tensors with fuzz data
        for (int i = 0; i < var_dim0 * var_dim1 && offset + 4 <= size; ++i) {
            var_flat(i) = *reinterpret_cast<const float*>(data + offset);
            accum_flat(i) = *reinterpret_cast<const float*>(data + offset + 4);
            offset += 8;
        }
        
        // Create indices tensor
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({indices_size}));
        auto indices_flat = indices_tensor.flat<int32_t>();
        
        for (int i = 0; i < indices_size && offset + 4 <= size; ++i) {
            indices_flat(i) = (*reinterpret_cast<const int32_t*>(data + offset)) % var_dim0;
            offset += 4;
        }
        
        // Create grad tensor
        tensorflow::Tensor grad_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({indices_size, var_dim1}));
        auto grad_flat = grad_tensor.flat<float>();
        
        for (int i = 0; i < indices_size * var_dim1 && offset + 4 <= size; ++i) {
            grad_flat(i) = *reinterpret_cast<const float*>(data + offset);
            offset += 4;
        }
        
        // Create learning rate and momentum tensors
        tensorflow::Tensor lr_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        lr_tensor.scalar<float>()() = lr;
        
        tensorflow::Tensor momentum_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        momentum_tensor.scalar<float>()() = momentum;
        
        // Create resource handles (simplified approach)
        tensorflow::Tensor var_handle(tensorflow::DT_RESOURCE, tensorflow::TensorShape({}));
        tensorflow::Tensor accum_handle(tensorflow::DT_RESOURCE, tensorflow::TensorShape({}));
        
        // Create a simple graph with ResourceSparseApplyMomentum
        tensorflow::GraphDef graph_def;
        tensorflow::GraphDefBuilder builder(tensorflow::GraphDefBuilder::kFailImmediately);
        
        // Add placeholder nodes
        auto var_ph = tensorflow::ops::Placeholder(builder.opts().WithName("var").WithAttr("dtype", tensorflow::DT_RESOURCE));
        auto accum_ph = tensorflow::ops::Placeholder(builder.opts().WithName("accum").WithAttr("dtype", tensorflow::DT_RESOURCE));
        auto lr_ph = tensorflow::ops::Placeholder(builder.opts().WithName("lr").WithAttr("dtype", tensorflow::DT_FLOAT));
        auto grad_ph = tensorflow::ops::Placeholder(builder.opts().WithName("grad").WithAttr("dtype", tensorflow::DT_FLOAT));
        auto indices_ph = tensorflow::ops::Placeholder(builder.opts().WithName("indices").WithAttr("dtype", tensorflow::DT_INT32));
        auto momentum_ph = tensorflow::ops::Placeholder(builder.opts().WithName("momentum").WithAttr("dtype", tensorflow::DT_FLOAT));
        
        // Add ResourceSparseApplyMomentum operation
        auto momentum_op = tensorflow::ops::UnaryOp("ResourceSparseApplyMomentum", var_ph,
                                                   builder.opts().WithName("momentum_op")
                                                   .WithAttr("T", tensorflow::DT_FLOAT)
                                                   .WithAttr("Tindices", tensorflow::DT_INT32)
                                                   .WithAttr("use_locking", use_locking)
                                                   .WithAttr("use_nesterov", use_nesterov));
        
        tensorflow::Status status = builder.ToGraphDef(&graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Create and run session (simplified)
        status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Prepare feed dict
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"var:0", var_handle},
            {"accum:0", accum_handle},
            {"lr:0", lr_tensor},
            {"grad:0", grad_tensor},
            {"indices:0", indices_tensor},
            {"momentum:0", momentum_tensor}
        };
        
        std::vector<std::string> output_names = {"momentum_op:0"};
        std::vector<tensorflow::Tensor> outputs;
        
        // Run the operation
        status = session->Run(inputs, output_names, {}, &outputs);
        
        // Clean up
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}