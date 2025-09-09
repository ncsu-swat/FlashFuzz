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
        offset++;
        int32_t indices_size = (data[offset] % 10) + 1;
        offset++;
        
        float lr = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        float l1 = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        float l2 = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        float l2_shrinkage = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        float lr_power = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        
        if (offset + var_dim * sizeof(float) * 3 + indices_size * sizeof(int32_t) * 2 > size) {
            return 0;
        }
        
        // Create TensorFlow session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create input tensors
        tensorflow::Tensor var_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({var_dim}));
        tensorflow::Tensor accum_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({var_dim}));
        tensorflow::Tensor linear_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({var_dim}));
        tensorflow::Tensor grad_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({indices_size}));
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({indices_size}));
        tensorflow::Tensor lr_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor l1_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor l2_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor l2_shrinkage_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor lr_power_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        
        // Fill tensors with fuzz data
        auto var_flat = var_tensor.flat<float>();
        auto accum_flat = accum_tensor.flat<float>();
        auto linear_flat = linear_tensor.flat<float>();
        auto grad_flat = grad_tensor.flat<float>();
        auto indices_flat = indices_tensor.flat<int32_t>();
        
        for (int i = 0; i < var_dim; i++) {
            var_flat(i) = *reinterpret_cast<const float*>(data + offset);
            offset += sizeof(float);
        }
        
        for (int i = 0; i < var_dim; i++) {
            accum_flat(i) = std::abs(*reinterpret_cast<const float*>(data + offset)) + 1e-6f;
            offset += sizeof(float);
        }
        
        for (int i = 0; i < var_dim; i++) {
            linear_flat(i) = *reinterpret_cast<const float*>(data + offset);
            offset += sizeof(float);
        }
        
        for (int i = 0; i < indices_size; i++) {
            grad_flat(i) = *reinterpret_cast<const float*>(data + offset);
            offset += sizeof(float);
        }
        
        for (int i = 0; i < indices_size; i++) {
            indices_flat(i) = std::abs(*reinterpret_cast<const int32_t*>(data + offset)) % var_dim;
            offset += sizeof(int32_t);
        }
        
        lr_tensor.scalar<float>()() = lr;
        l1_tensor.scalar<float>()() = std::abs(l1);
        l2_tensor.scalar<float>()() = std::abs(l2);
        l2_shrinkage_tensor.scalar<float>()() = std::abs(l2_shrinkage);
        lr_power_tensor.scalar<float>()() = lr_power;
        
        // Create graph definition
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        node_def->set_name("sparse_apply_ftrl_v2");
        node_def->set_op("SparseApplyFtrlV2");
        
        node_def->add_input("var");
        node_def->add_input("accum");
        node_def->add_input("linear");
        node_def->add_input("grad");
        node_def->add_input("indices");
        node_def->add_input("lr");
        node_def->add_input("l1");
        node_def->add_input("l2");
        node_def->add_input("l2_shrinkage");
        node_def->add_input("lr_power");
        
        (*node_def->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
        (*node_def->mutable_attr())["Tindices"].set_type(tensorflow::DT_INT32);
        (*node_def->mutable_attr())["use_locking"].set_b(false);
        (*node_def->mutable_attr())["multiply_linear_by_lr"].set_b(false);
        
        // Add placeholder nodes
        auto add_placeholder = [&](const std::string& name, tensorflow::DataType dtype, const tensorflow::TensorShape& shape) {
            tensorflow::NodeDef* placeholder = graph_def.add_node();
            placeholder->set_name(name);
            placeholder->set_op("Placeholder");
            (*placeholder->mutable_attr())["dtype"].set_type(dtype);
            auto shape_proto = (*placeholder->mutable_attr())["shape"].mutable_shape();
            for (int i = 0; i < shape.dims(); i++) {
                shape_proto->add_dim()->set_size(shape.dim_size(i));
            }
        };
        
        add_placeholder("var", tensorflow::DT_FLOAT, var_tensor.shape());
        add_placeholder("accum", tensorflow::DT_FLOAT, accum_tensor.shape());
        add_placeholder("linear", tensorflow::DT_FLOAT, linear_tensor.shape());
        add_placeholder("grad", tensorflow::DT_FLOAT, grad_tensor.shape());
        add_placeholder("indices", tensorflow::DT_INT32, indices_tensor.shape());
        add_placeholder("lr", tensorflow::DT_FLOAT, lr_tensor.shape());
        add_placeholder("l1", tensorflow::DT_FLOAT, l1_tensor.shape());
        add_placeholder("l2", tensorflow::DT_FLOAT, l2_tensor.shape());
        add_placeholder("l2_shrinkage", tensorflow::DT_FLOAT, l2_shrinkage_tensor.shape());
        add_placeholder("lr_power", tensorflow::DT_FLOAT, lr_power_tensor.shape());
        
        // Create session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"var", var_tensor},
            {"accum", accum_tensor},
            {"linear", linear_tensor},
            {"grad", grad_tensor},
            {"indices", indices_tensor},
            {"lr", lr_tensor},
            {"l1", l1_tensor},
            {"l2", l2_tensor},
            {"l2_shrinkage", l2_shrinkage_tensor},
            {"lr_power", lr_power_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"sparse_apply_ftrl_v2"}, {}, &outputs);
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}