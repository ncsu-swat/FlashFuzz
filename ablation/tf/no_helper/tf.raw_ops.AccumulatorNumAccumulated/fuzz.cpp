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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < sizeof(uint32_t)) {
            return 0;
        }
        
        // Extract handle string length
        uint32_t handle_len;
        memcpy(&handle_len, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        // Limit handle length to prevent excessive memory usage
        handle_len = handle_len % 1024;
        
        if (offset + handle_len > size) {
            return 0;
        }
        
        // Create handle string
        std::string handle_str(reinterpret_cast<const char*>(data + offset), handle_len);
        offset += handle_len;
        
        // Create TensorFlow session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create graph definition
        tensorflow::GraphDef graph_def;
        
        // Add ConditionalAccumulator operation first to create a valid handle
        auto* accumulator_node = graph_def.add_node();
        accumulator_node->set_name("test_accumulator");
        accumulator_node->set_op("ConditionalAccumulator");
        (*accumulator_node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        (*accumulator_node->mutable_attr())["shape"].mutable_shape();
        (*accumulator_node->mutable_attr())["container"].set_s("");
        (*accumulator_node->mutable_attr())["shared_name"].set_s("test_shared_accumulator");
        
        // Add Const node for handle input
        auto* const_node = graph_def.add_node();
        const_node->set_name("handle_input");
        const_node->set_op("Const");
        (*const_node->mutable_attr())["dtype"].set_type(tensorflow::DT_STRING);
        auto* tensor_proto = (*const_node->mutable_attr())["value"].mutable_tensor();
        tensor_proto->set_dtype(tensorflow::DT_STRING);
        tensor_proto->mutable_tensor_shape();
        tensor_proto->add_string_val(handle_str);
        
        // Add AccumulatorNumAccumulated operation
        auto* num_accumulated_node = graph_def.add_node();
        num_accumulated_node->set_name("num_accumulated");
        num_accumulated_node->set_op("AccumulatorNumAccumulated");
        num_accumulated_node->add_input("handle_input");
        
        // Create session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({}, {"num_accumulated"}, {}, &outputs);
        
        // Clean up
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}