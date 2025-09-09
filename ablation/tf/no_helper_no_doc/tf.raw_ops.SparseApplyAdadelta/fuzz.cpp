#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/framework/fake_input.h>
#include <tensorflow/core/kernels/ops_testutil.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions and parameters from fuzz input
        int32_t var_dim = (data[offset] % 10) + 1;
        offset += 1;
        int32_t accum_dim = var_dim;
        int32_t accum_update_dim = var_dim;
        int32_t grad_dim = var_dim;
        
        int32_t indices_size = (data[offset] % 5) + 1;
        offset += 1;
        
        // Extract learning rate, rho, epsilon
        float lr = *reinterpret_cast<const float*>(data + offset);
        offset += 4;
        float rho = *reinterpret_cast<const float*>(data + offset);
        offset += 4;
        float epsilon = *reinterpret_cast<const float*>(data + offset);
        offset += 4;
        
        bool use_locking = (data[offset] % 2) == 1;
        offset += 1;
        
        if (offset + var_dim * 4 + accum_dim * 4 + accum_update_dim * 4 + 
            grad_dim * 4 + indices_size * 4 > size) {
            return 0;
        }
        
        // Clamp values to reasonable ranges
        lr = std::max(-10.0f, std::min(10.0f, lr));
        rho = std::max(0.0f, std::min(1.0f, rho));
        epsilon = std::max(1e-8f, std::min(1.0f, std::abs(epsilon)));
        
        // Create TensorFlow session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create graph
        tensorflow::GraphDef graph_def;
        
        // Create input tensors
        tensorflow::Tensor var_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({var_dim}));
        tensorflow::Tensor accum_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({accum_dim}));
        tensorflow::Tensor accum_update_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({accum_update_dim}));
        tensorflow::Tensor lr_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor rho_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor epsilon_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor grad_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({indices_size}));
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({indices_size}));
        
        // Fill tensors with fuzz data
        auto var_flat = var_tensor.flat<float>();
        for (int i = 0; i < var_dim && offset + 4 <= size; ++i) {
            var_flat(i) = *reinterpret_cast<const float*>(data + offset);
            offset += 4;
        }
        
        auto accum_flat = accum_tensor.flat<float>();
        for (int i = 0; i < accum_dim && offset + 4 <= size; ++i) {
            accum_flat(i) = std::max(0.0f, *reinterpret_cast<const float*>(data + offset));
            offset += 4;
        }
        
        auto accum_update_flat = accum_update_tensor.flat<float>();
        for (int i = 0; i < accum_update_dim && offset + 4 <= size; ++i) {
            accum_update_flat(i) = std::max(0.0f, *reinterpret_cast<const float*>(data + offset));
            offset += 4;
        }
        
        lr_tensor.scalar<float>()() = lr;
        rho_tensor.scalar<float>()() = rho;
        epsilon_tensor.scalar<float>()() = epsilon;
        
        auto grad_flat = grad_tensor.flat<float>();
        for (int i = 0; i < indices_size && offset + 4 <= size; ++i) {
            grad_flat(i) = *reinterpret_cast<const float*>(data + offset);
            offset += 4;
        }
        
        auto indices_flat = indices_tensor.flat<int32_t>();
        for (int i = 0; i < indices_size && offset + 4 <= size; ++i) {
            indices_flat(i) = std::abs(*reinterpret_cast<const int32_t*>(data + offset)) % var_dim;
            offset += 4;
        }
        
        // Create node definition
        tensorflow::NodeDef node_def;
        node_def.set_name("sparse_apply_adadelta");
        node_def.set_op("SparseApplyAdadelta");
        node_def.add_input("var");
        node_def.add_input("accum");
        node_def.add_input("accum_update");
        node_def.add_input("lr");
        node_def.add_input("rho");
        node_def.add_input("epsilon");
        node_def.add_input("grad");
        node_def.add_input("indices");
        
        auto& attr_map = *node_def.mutable_attr();
        attr_map["T"].set_type(tensorflow::DT_FLOAT);
        attr_map["Tindices"].set_type(tensorflow::DT_INT32);
        attr_map["use_locking"].set_b(use_locking);
        
        // Add placeholder nodes for inputs
        tensorflow::NodeDef var_node, accum_node, accum_update_node, lr_node, rho_node, epsilon_node, grad_node, indices_node;
        
        var_node.set_name("var");
        var_node.set_op("Placeholder");
        (*var_node.mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        
        accum_node.set_name("accum");
        accum_node.set_op("Placeholder");
        (*accum_node.mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        
        accum_update_node.set_name("accum_update");
        accum_update_node.set_op("Placeholder");
        (*accum_update_node.mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        
        lr_node.set_name("lr");
        lr_node.set_op("Placeholder");
        (*lr_node.mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        
        rho_node.set_name("rho");
        rho_node.set_op("Placeholder");
        (*rho_node.mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        
        epsilon_node.set_name("epsilon");
        epsilon_node.set_op("Placeholder");
        (*epsilon_node.mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        
        grad_node.set_name("grad");
        grad_node.set_op("Placeholder");
        (*grad_node.mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        
        indices_node.set_name("indices");
        indices_node.set_op("Placeholder");
        (*indices_node.mutable_attr())["dtype"].set_type(tensorflow::DT_INT32);
        
        // Add nodes to graph
        *graph_def.add_node() = var_node;
        *graph_def.add_node() = accum_node;
        *graph_def.add_node() = accum_update_node;
        *graph_def.add_node() = lr_node;
        *graph_def.add_node() = rho_node;
        *graph_def.add_node() = epsilon_node;
        *graph_def.add_node() = grad_node;
        *graph_def.add_node() = indices_node;
        *graph_def.add_node() = node_def;
        
        // Create session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"var", var_tensor},
            {"accum", accum_tensor},
            {"accum_update", accum_update_tensor},
            {"lr", lr_tensor},
            {"rho", rho_tensor},
            {"epsilon", epsilon_tensor},
            {"grad", grad_tensor},
            {"indices", indices_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {"sparse_apply_adadelta"};
        
        status = session->Run(inputs, output_names, {}, &outputs);
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}