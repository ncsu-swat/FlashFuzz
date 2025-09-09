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
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/common_runtime/direct_session.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions and data type
        uint32_t dim1 = *reinterpret_cast<const uint32_t*>(data + offset) % 10 + 1;
        offset += 4;
        uint32_t dim2 = *reinterpret_cast<const uint32_t*>(data + offset) % 10 + 1;
        offset += 4;
        uint32_t data_type = *reinterpret_cast<const uint32_t*>(data + offset) % 3;
        offset += 4;
        uint32_t use_locking = *reinterpret_cast<const uint32_t*>(data + offset) % 2;
        offset += 4;
        
        tensorflow::DataType dtype;
        size_t element_size;
        switch (data_type) {
            case 0:
                dtype = tensorflow::DT_FLOAT;
                element_size = sizeof(float);
                break;
            case 1:
                dtype = tensorflow::DT_DOUBLE;
                element_size = sizeof(double);
                break;
            case 2:
                dtype = tensorflow::DT_INT32;
                element_size = sizeof(int32_t);
                break;
            default:
                dtype = tensorflow::DT_FLOAT;
                element_size = sizeof(float);
                break;
        }
        
        tensorflow::TensorShape shape({static_cast<int64_t>(dim1), static_cast<int64_t>(dim2)});
        size_t total_elements = dim1 * dim2;
        size_t required_bytes = total_elements * element_size * 2; // for ref and value tensors
        
        if (offset + required_bytes > size) return 0;
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create variable tensor (ref)
        tensorflow::Tensor ref_tensor(dtype, shape);
        tensorflow::Tensor value_tensor(dtype, shape);
        
        // Fill tensors with fuzz data
        if (dtype == tensorflow::DT_FLOAT) {
            auto ref_flat = ref_tensor.flat<float>();
            auto value_flat = value_tensor.flat<float>();
            for (size_t i = 0; i < total_elements && offset + sizeof(float) <= size; ++i) {
                float val = *reinterpret_cast<const float*>(data + offset);
                if (std::isfinite(val)) {
                    ref_flat(i) = val;
                } else {
                    ref_flat(i) = 1.0f;
                }
                offset += sizeof(float);
            }
            for (size_t i = 0; i < total_elements && offset + sizeof(float) <= size; ++i) {
                float val = *reinterpret_cast<const float*>(data + offset);
                if (std::isfinite(val)) {
                    value_flat(i) = val;
                } else {
                    value_flat(i) = 1.0f;
                }
                offset += sizeof(float);
            }
        } else if (dtype == tensorflow::DT_DOUBLE) {
            auto ref_flat = ref_tensor.flat<double>();
            auto value_flat = value_tensor.flat<double>();
            for (size_t i = 0; i < total_elements && offset + sizeof(double) <= size; ++i) {
                double val = *reinterpret_cast<const double*>(data + offset);
                if (std::isfinite(val)) {
                    ref_flat(i) = val;
                } else {
                    ref_flat(i) = 1.0;
                }
                offset += sizeof(double);
            }
            for (size_t i = 0; i < total_elements && offset + sizeof(double) <= size; ++i) {
                double val = *reinterpret_cast<const double*>(data + offset);
                if (std::isfinite(val)) {
                    value_flat(i) = val;
                } else {
                    value_flat(i) = 1.0;
                }
                offset += sizeof(double);
            }
        } else if (dtype == tensorflow::DT_INT32) {
            auto ref_flat = ref_tensor.flat<int32_t>();
            auto value_flat = value_tensor.flat<int32_t>();
            for (size_t i = 0; i < total_elements && offset + sizeof(int32_t) <= size; ++i) {
                ref_flat(i) = *reinterpret_cast<const int32_t*>(data + offset);
                offset += sizeof(int32_t);
            }
            for (size_t i = 0; i < total_elements && offset + sizeof(int32_t) <= size; ++i) {
                value_flat(i) = *reinterpret_cast<const int32_t*>(data + offset);
                offset += sizeof(int32_t);
            }
        }
        
        // Create graph with AssignSub operation
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* variable_node = graph_def.add_node();
        variable_node->set_name("variable");
        variable_node->set_op("Variable");
        (*variable_node->mutable_attr())["dtype"].set_type(dtype);
        (*variable_node->mutable_attr())["shape"].mutable_shape()->CopyFrom(shape.AsProto());
        
        tensorflow::NodeDef* assign_node = graph_def.add_node();
        assign_node->set_name("assign");
        assign_node->set_op("Assign");
        assign_node->add_input("variable");
        assign_node->add_input("init_value");
        (*assign_node->mutable_attr())["T"].set_type(dtype);
        (*assign_node->mutable_attr())["use_locking"].set_b(false);
        (*assign_node->mutable_attr())["validate_shape"].set_b(true);
        
        tensorflow::NodeDef* init_node = graph_def.add_node();
        init_node->set_name("init_value");
        init_node->set_op("Const");
        (*init_node->mutable_attr())["dtype"].set_type(dtype);
        ref_tensor.AsProtoTensorContent((*init_node->mutable_attr())["value"].mutable_tensor());
        
        tensorflow::NodeDef* value_node = graph_def.add_node();
        value_node->set_name("value");
        value_node->set_op("Const");
        (*value_node->mutable_attr())["dtype"].set_type(dtype);
        value_tensor.AsProtoTensorContent((*value_node->mutable_attr())["value"].mutable_tensor());
        
        tensorflow::NodeDef* assign_sub_node = graph_def.add_node();
        assign_sub_node->set_name("assign_sub");
        assign_sub_node->set_op("AssignSub");
        assign_sub_node->add_input("variable");
        assign_sub_node->add_input("value");
        (*assign_sub_node->mutable_attr())["T"].set_type(dtype);
        (*assign_sub_node->mutable_attr())["use_locking"].set_b(use_locking != 0);
        
        // Create and run session
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Initialize variable
        std::vector<tensorflow::Tensor> init_outputs;
        status = session->Run({}, {"assign"}, {}, &init_outputs);
        if (!status.ok()) {
            return 0;
        }
        
        // Run AssignSub
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({}, {"assign_sub"}, {}, &outputs);
        if (!status.ok()) {
            return 0;
        }
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}