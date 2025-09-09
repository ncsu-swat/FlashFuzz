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
        int32_t var_dim0 = *reinterpret_cast<const int32_t*>(data + offset) % 100 + 1;
        offset += 4;
        int32_t var_dim1 = *reinterpret_cast<const int32_t*>(data + offset) % 100 + 1;
        offset += 4;
        int32_t num_indices = *reinterpret_cast<const int32_t*>(data + offset) % 10 + 1;
        offset += 4;
        
        float lr = *reinterpret_cast<const float*>(data + offset);
        offset += 4;
        float l1 = *reinterpret_cast<const float*>(data + offset);
        offset += 4;
        float l2 = *reinterpret_cast<const float*>(data + offset);
        offset += 4;
        int64_t global_step = *reinterpret_cast<const int64_t*>(data + offset);
        offset += 8;
        
        // Clamp values to reasonable ranges
        lr = std::max(0.001f, std::min(1.0f, std::abs(lr)));
        l1 = std::max(0.0f, std::min(1.0f, std::abs(l1)));
        l2 = std::max(0.0f, std::min(1.0f, std::abs(l2)));
        global_step = std::max(1LL, std::abs(global_step) % 1000 + 1);
        
        // Create tensors
        tensorflow::Tensor var(tensorflow::DT_FLOAT, tensorflow::TensorShape({var_dim0, var_dim1}));
        tensorflow::Tensor gradient_accumulator(tensorflow::DT_FLOAT, tensorflow::TensorShape({var_dim0, var_dim1}));
        tensorflow::Tensor gradient_squared_accumulator(tensorflow::DT_FLOAT, tensorflow::TensorShape({var_dim0, var_dim1}));
        tensorflow::Tensor grad(tensorflow::DT_FLOAT, tensorflow::TensorShape({num_indices, var_dim1}));
        tensorflow::Tensor indices(tensorflow::DT_INT32, tensorflow::TensorShape({num_indices}));
        tensorflow::Tensor lr_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor l1_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor l2_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor global_step_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        
        // Initialize tensors with fuzz data
        auto var_flat = var.flat<float>();
        auto grad_acc_flat = gradient_accumulator.flat<float>();
        auto grad_sq_acc_flat = gradient_squared_accumulator.flat<float>();
        auto grad_flat = grad.flat<float>();
        auto indices_flat = indices.flat<int32_t>();
        
        size_t remaining_size = size - offset;
        size_t float_count = var_flat.size() + grad_acc_flat.size() + grad_sq_acc_flat.size() + grad_flat.size();
        
        if (remaining_size < float_count * sizeof(float) + num_indices * sizeof(int32_t)) {
            // Fill with default values if not enough data
            for (int i = 0; i < var_flat.size(); ++i) {
                var_flat(i) = 0.1f;
                grad_acc_flat(i) = 0.0f;
                grad_sq_acc_flat(i) = 0.1f;
            }
            for (int i = 0; i < grad_flat.size(); ++i) {
                grad_flat(i) = 0.01f;
            }
            for (int i = 0; i < num_indices; ++i) {
                indices_flat(i) = i % var_dim0;
            }
        } else {
            // Use fuzz data
            for (int i = 0; i < var_flat.size() && offset < size - 4; ++i) {
                var_flat(i) = *reinterpret_cast<const float*>(data + offset);
                offset += 4;
            }
            for (int i = 0; i < grad_acc_flat.size() && offset < size - 4; ++i) {
                grad_acc_flat(i) = *reinterpret_cast<const float*>(data + offset);
                offset += 4;
            }
            for (int i = 0; i < grad_sq_acc_flat.size() && offset < size - 4; ++i) {
                grad_sq_acc_flat(i) = std::abs(*reinterpret_cast<const float*>(data + offset)) + 0.001f;
                offset += 4;
            }
            for (int i = 0; i < grad_flat.size() && offset < size - 4; ++i) {
                grad_flat(i) = *reinterpret_cast<const float*>(data + offset);
                offset += 4;
            }
            for (int i = 0; i < num_indices && offset < size - 4; ++i) {
                indices_flat(i) = std::abs(*reinterpret_cast<const int32_t*>(data + offset)) % var_dim0;
                offset += 4;
            }
        }
        
        lr_tensor.scalar<float>()() = lr;
        l1_tensor.scalar<float>()() = l1;
        l2_tensor.scalar<float>()() = l2;
        global_step_tensor.scalar<int64_t>()() = global_step;
        
        // Create session and graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        
        node_def->set_name("sparse_apply_adagrad_da");
        node_def->set_op("SparseApplyAdagradDA");
        node_def->add_input("var");
        node_def->add_input("gradient_accumulator");
        node_def->add_input("gradient_squared_accumulator");
        node_def->add_input("grad");
        node_def->add_input("indices");
        node_def->add_input("lr");
        node_def->add_input("l1");
        node_def->add_input("l2");
        node_def->add_input("global_step");
        
        // Add input nodes
        for (const auto& input_name : {"var", "gradient_accumulator", "gradient_squared_accumulator", 
                                       "grad", "indices", "lr", "l1", "l2", "global_step"}) {
            tensorflow::NodeDef* input_node = graph_def.add_node();
            input_node->set_name(input_name);
            input_node->set_op("Placeholder");
            if (std::string(input_name) == "indices") {
                input_node->mutable_attr()->insert({"dtype", tensorflow::AttrValue()});
                input_node->mutable_attr()->at("dtype").set_type(tensorflow::DT_INT32);
            } else if (std::string(input_name) == "global_step") {
                input_node->mutable_attr()->insert({"dtype", tensorflow::AttrValue()});
                input_node->mutable_attr()->at("dtype").set_type(tensorflow::DT_INT64);
            } else {
                input_node->mutable_attr()->insert({"dtype", tensorflow::AttrValue()});
                input_node->mutable_attr()->at("dtype").set_type(tensorflow::DT_FLOAT);
            }
        }
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the operation
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"var", var},
            {"gradient_accumulator", gradient_accumulator},
            {"gradient_squared_accumulator", gradient_squared_accumulator},
            {"grad", grad},
            {"indices", indices},
            {"lr", lr_tensor},
            {"l1", l1_tensor},
            {"l2", l2_tensor},
            {"global_step", global_step_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"sparse_apply_adagrad_da"}, {}, &outputs);
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}