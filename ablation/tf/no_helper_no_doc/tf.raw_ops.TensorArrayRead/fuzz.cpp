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
        
        if (size < 16) return 0;
        
        // Extract handle value
        int32_t handle_value = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        // Extract index value
        int32_t index_value = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        // Extract flow_in value
        float flow_in_value = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        
        // Extract dtype
        int32_t dtype_raw = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        // Clamp dtype to valid range
        tensorflow::DataType dtype = static_cast<tensorflow::DataType>(
            (dtype_raw % 23) + 1); // DT_FLOAT = 1, limit to reasonable range
        
        // Create tensors
        tensorflow::Tensor handle_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        handle_tensor.scalar<tensorflow::tstring>()() = tensorflow::tstring(
            reinterpret_cast<const char*>(&handle_value), sizeof(handle_value));
        
        tensorflow::Tensor index_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        index_tensor.scalar<int32_t>()() = std::abs(index_value) % 1000; // Limit index range
        
        tensorflow::Tensor flow_in_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        flow_in_tensor.scalar<float>()() = flow_in_value;
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        node_def->set_name("tensor_array_read");
        node_def->set_op("TensorArrayReadV3");
        node_def->add_input("handle:0");
        node_def->add_input("index:0");
        node_def->add_input("flow_in:0");
        
        // Set dtype attribute
        tensorflow::AttrValue dtype_attr;
        dtype_attr.set_type(dtype);
        (*node_def->mutable_attr())["dtype"] = dtype_attr;
        
        // Add placeholder nodes for inputs
        tensorflow::NodeDef* handle_node = graph_def.add_node();
        handle_node->set_name("handle");
        handle_node->set_op("Placeholder");
        tensorflow::AttrValue handle_dtype_attr;
        handle_dtype_attr.set_type(tensorflow::DT_STRING);
        (*handle_node->mutable_attr())["dtype"] = handle_dtype_attr;
        
        tensorflow::NodeDef* index_node = graph_def.add_node();
        index_node->set_name("index");
        index_node->set_op("Placeholder");
        tensorflow::AttrValue index_dtype_attr;
        index_dtype_attr.set_type(tensorflow::DT_INT32);
        (*index_node->mutable_attr())["dtype"] = index_dtype_attr;
        
        tensorflow::NodeDef* flow_in_node = graph_def.add_node();
        flow_in_node->set_name("flow_in");
        flow_in_node->set_op("Placeholder");
        tensorflow::AttrValue flow_in_dtype_attr;
        flow_in_dtype_attr.set_type(tensorflow::DT_FLOAT);
        (*flow_in_node->mutable_attr())["dtype"] = flow_in_dtype_attr;
        
        // Create session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0; // Skip invalid graphs
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"handle:0", handle_tensor},
            {"index:0", index_tensor},
            {"flow_in:0", flow_in_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {"tensor_array_read:0"};
        
        // Run the operation (may fail, which is expected for fuzzing)
        session->Run(inputs, output_names, {}, &outputs);
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}