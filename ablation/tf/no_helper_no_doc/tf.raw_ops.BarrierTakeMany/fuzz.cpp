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
        
        if (size < 20) return 0;
        
        // Extract handle string length
        if (offset + 4 > size) return 0;
        uint32_t handle_len = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        handle_len = handle_len % 256; // Limit handle length
        
        if (offset + handle_len > size) return 0;
        std::string handle(reinterpret_cast<const char*>(data + offset), handle_len);
        offset += handle_len;
        
        // Extract num_elements
        if (offset + 4 > size) return 0;
        int32_t num_elements = *reinterpret_cast<const int32_t*>(data + offset);
        offset += 4;
        num_elements = std::abs(num_elements) % 10 + 1; // Limit to reasonable range
        
        // Extract allow_small_batch
        if (offset + 1 > size) return 0;
        bool allow_small_batch = (data[offset] % 2) == 1;
        offset += 1;
        
        // Extract wait_for_incomplete
        if (offset + 1 > size) return 0;
        bool wait_for_incomplete = (data[offset] % 2) == 1;
        offset += 1;
        
        // Extract timeout_ms
        if (offset + 8 > size) return 0;
        int64_t timeout_ms = *reinterpret_cast<const int64_t*>(data + offset);
        offset += 8;
        timeout_ms = std::abs(timeout_ms) % 10000; // Limit timeout
        
        // Create TensorFlow session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create graph def
        tensorflow::GraphDef graph_def;
        
        // Create handle tensor
        tensorflow::NodeDef handle_node;
        handle_node.set_name("handle");
        handle_node.set_op("Const");
        tensorflow::AttrValue handle_attr;
        handle_attr.mutable_tensor()->set_dtype(tensorflow::DT_STRING);
        handle_attr.mutable_tensor()->mutable_tensor_shape();
        handle_attr.mutable_tensor()->add_string_val(handle);
        (*handle_node.mutable_attr())["value"] = handle_attr;
        (*handle_node.mutable_attr())["dtype"].set_type(tensorflow::DT_STRING);
        *graph_def.add_node() = handle_node;
        
        // Create num_elements tensor
        tensorflow::NodeDef num_elements_node;
        num_elements_node.set_name("num_elements");
        num_elements_node.set_op("Const");
        tensorflow::AttrValue num_elements_attr;
        num_elements_attr.mutable_tensor()->set_dtype(tensorflow::DT_INT32);
        num_elements_attr.mutable_tensor()->mutable_tensor_shape();
        num_elements_attr.mutable_tensor()->add_int_val(num_elements);
        (*num_elements_node.mutable_attr())["value"] = num_elements_attr;
        (*num_elements_node.mutable_attr())["dtype"].set_type(tensorflow::DT_INT32);
        *graph_def.add_node() = num_elements_node;
        
        // Create BarrierTakeMany node
        tensorflow::NodeDef barrier_node;
        barrier_node.set_name("barrier_take_many");
        barrier_node.set_op("BarrierTakeMany");
        barrier_node.add_input("handle");
        barrier_node.add_input("num_elements");
        
        // Set component_types attribute (using common types)
        tensorflow::AttrValue component_types_attr;
        component_types_attr.mutable_list()->add_type(tensorflow::DT_FLOAT);
        component_types_attr.mutable_list()->add_type(tensorflow::DT_INT32);
        (*barrier_node.mutable_attr())["component_types"] = component_types_attr;
        
        (*barrier_node.mutable_attr())["allow_small_batch"].set_b(allow_small_batch);
        (*barrier_node.mutable_attr())["wait_for_incomplete"].set_b(wait_for_incomplete);
        (*barrier_node.mutable_attr())["timeout_ms"].set_i(timeout_ms);
        
        *graph_def.add_node() = barrier_node;
        
        // Create the session with the graph
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0; // Ignore creation failures in fuzzing
        }
        
        // Run the operation (this will likely fail due to barrier not being initialized, but that's expected in fuzzing)
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({}, {"barrier_take_many:0", "barrier_take_many:1", "barrier_take_many:2"}, {}, &outputs);
        
        // Close session
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}