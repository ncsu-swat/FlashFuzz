#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/framework/node_def.pb.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/common_runtime/direct_session.h>
#include <tensorflow/core/framework/allocator.h>
#include <tensorflow/core/framework/device_factory.h>
#include <tensorflow/core/common_runtime/device_mgr.h>
#include <tensorflow/core/common_runtime/device_factory.h>
#include <tensorflow/core/framework/op_def_builder.h>
#include <tensorflow/core/kernels/ops_util.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 64) return 0;
        
        // Extract dimensions and parameters from fuzz input
        int32_t dim1 = *reinterpret_cast<const int32_t*>(data + offset) % 10 + 1;
        offset += sizeof(int32_t);
        int32_t dim2 = *reinterpret_cast<const int32_t*>(data + offset) % 10 + 1;
        offset += sizeof(int32_t);
        
        // Extract scalar values
        float lr_val = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        float l1_val = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        float l2_val = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        float l2_shrinkage_val = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        float lr_power_val = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        
        bool use_locking = (data[offset] % 2) == 1;
        offset++;
        bool multiply_linear_by_lr = (data[offset] % 2) == 1;
        offset++;
        
        // Ensure we have enough data for tensor values
        size_t required_size = dim1 * dim2 * sizeof(float) * 4; // var, accum, linear, grad
        if (offset + required_size > size) return 0;
        
        tensorflow::TensorShape shape({dim1, dim2});
        tensorflow::TensorShape scalar_shape({});
        
        // Create tensors
        tensorflow::Tensor var_tensor(tensorflow::DT_FLOAT, shape);
        tensorflow::Tensor accum_tensor(tensorflow::DT_FLOAT, shape);
        tensorflow::Tensor linear_tensor(tensorflow::DT_FLOAT, shape);
        tensorflow::Tensor grad_tensor(tensorflow::DT_FLOAT, shape);
        tensorflow::Tensor lr_tensor(tensorflow::DT_FLOAT, scalar_shape);
        tensorflow::Tensor l1_tensor(tensorflow::DT_FLOAT, scalar_shape);
        tensorflow::Tensor l2_tensor(tensorflow::DT_FLOAT, scalar_shape);
        tensorflow::Tensor l2_shrinkage_tensor(tensorflow::DT_FLOAT, scalar_shape);
        tensorflow::Tensor lr_power_tensor(tensorflow::DT_FLOAT, scalar_shape);
        
        // Fill tensors with fuzz data
        auto var_flat = var_tensor.flat<float>();
        auto accum_flat = accum_tensor.flat<float>();
        auto linear_flat = linear_tensor.flat<float>();
        auto grad_flat = grad_tensor.flat<float>();
        
        for (int i = 0; i < dim1 * dim2; ++i) {
            if (offset + sizeof(float) <= size) {
                var_flat(i) = *reinterpret_cast<const float*>(data + offset);
                offset += sizeof(float);
            } else {
                var_flat(i) = 1.0f;
            }
        }
        
        for (int i = 0; i < dim1 * dim2; ++i) {
            if (offset + sizeof(float) <= size) {
                accum_flat(i) = std::abs(*reinterpret_cast<const float*>(data + offset)) + 0.1f; // Ensure positive
                offset += sizeof(float);
            } else {
                accum_flat(i) = 1.0f;
            }
        }
        
        for (int i = 0; i < dim1 * dim2; ++i) {
            if (offset + sizeof(float) <= size) {
                linear_flat(i) = *reinterpret_cast<const float*>(data + offset);
                offset += sizeof(float);
            } else {
                linear_flat(i) = 0.0f;
            }
        }
        
        for (int i = 0; i < dim1 * dim2; ++i) {
            if (offset + sizeof(float) <= size) {
                grad_flat(i) = *reinterpret_cast<const float*>(data + offset);
                offset += sizeof(float);
            } else {
                grad_flat(i) = 0.1f;
            }
        }
        
        // Set scalar values
        lr_tensor.scalar<float>()() = std::abs(lr_val) + 0.001f; // Ensure positive learning rate
        l1_tensor.scalar<float>()() = std::abs(l1_val);
        l2_tensor.scalar<float>()() = std::abs(l2_val);
        l2_shrinkage_tensor.scalar<float>()() = std::abs(l2_shrinkage_val);
        lr_power_tensor.scalar<float>()() = lr_power_val;
        
        // Create session and graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        node_def->set_name("apply_ftrl_v2");
        node_def->set_op("ApplyFtrlV2");
        
        // Add inputs
        node_def->add_input("var");
        node_def->add_input("accum");
        node_def->add_input("linear");
        node_def->add_input("grad");
        node_def->add_input("lr");
        node_def->add_input("l1");
        node_def->add_input("l2");
        node_def->add_input("l2_shrinkage");
        node_def->add_input("lr_power");
        
        // Set attributes
        tensorflow::AttrValue use_locking_attr;
        use_locking_attr.set_b(use_locking);
        (*node_def->mutable_attr())["use_locking"] = use_locking_attr;
        
        tensorflow::AttrValue multiply_linear_by_lr_attr;
        multiply_linear_by_lr_attr.set_b(multiply_linear_by_lr);
        (*node_def->mutable_attr())["multiply_linear_by_lr"] = multiply_linear_by_lr_attr;
        
        tensorflow::AttrValue type_attr;
        type_attr.set_type(tensorflow::DT_FLOAT);
        (*node_def->mutable_attr())["T"] = type_attr;
        
        // Create placeholder nodes for inputs
        std::vector<std::string> input_names = {"var", "accum", "linear", "grad", "lr", "l1", "l2", "l2_shrinkage", "lr_power"};
        for (const auto& name : input_names) {
            tensorflow::NodeDef* placeholder = graph_def.add_node();
            placeholder->set_name(name);
            placeholder->set_op("Placeholder");
            tensorflow::AttrValue dtype_attr;
            dtype_attr.set_type(tensorflow::DT_FLOAT);
            (*placeholder->mutable_attr())["dtype"] = dtype_attr;
        }
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Prepare inputs
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"var", var_tensor},
            {"accum", accum_tensor},
            {"linear", linear_tensor},
            {"grad", grad_tensor},
            {"lr", lr_tensor},
            {"l1", l1_tensor},
            {"l2", l2_tensor},
            {"l2_shrinkage", l2_shrinkage_tensor},
            {"lr_power", lr_power_tensor}
        };
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"apply_ftrl_v2"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Operation succeeded, check output tensor
            const auto& output = outputs[0];
            if (output.shape() == shape && output.dtype() == tensorflow::DT_FLOAT) {
                // Verify output is finite
                auto output_flat = output.flat<float>();
                for (int i = 0; i < output_flat.size(); ++i) {
                    if (!std::isfinite(output_flat(i))) {
                        break;
                    }
                }
            }
        }
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}