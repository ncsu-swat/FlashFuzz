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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) {
            return 0;
        }
        
        // Extract handle length
        if (offset + sizeof(uint32_t) > size) return 0;
        uint32_t handle_len = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += sizeof(uint32_t);
        
        // Limit handle length to prevent excessive memory allocation
        handle_len = handle_len % 256;
        if (handle_len == 0) handle_len = 1;
        
        if (offset + handle_len > size) return 0;
        
        // Create handle string
        std::string handle(reinterpret_cast<const char*>(data + offset), handle_len);
        offset += handle_len;
        
        // Extract dtype
        if (offset + sizeof(uint32_t) > size) return 0;
        uint32_t dtype_val = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += sizeof(uint32_t);
        
        tensorflow::DataType dtype = static_cast<tensorflow::DataType>(dtype_val % 23 + 1); // Valid TF dtypes
        
        // Create a simple graph with GetSessionTensor operation
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node = graph_def.add_node();
        node->set_name("get_session_tensor");
        node->set_op("GetSessionTensor");
        
        // Set handle attribute
        tensorflow::AttrValue handle_attr;
        handle_attr.set_s(handle);
        (*node->mutable_attr())["handle"] = handle_attr;
        
        // Set dtype attribute
        tensorflow::AttrValue dtype_attr;
        dtype_attr.set_type(dtype);
        (*node->mutable_attr())["dtype"] = dtype_attr;
        
        // Create session options
        tensorflow::SessionOptions session_options;
        
        // Create session
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(session_options));
        if (!session) {
            return 0;
        }
        
        // Create the session (this will likely fail, but we're testing the operation creation)
        tensorflow::Status status = session->Create(graph_def);
        
        // Try to run the operation (expected to fail in most cases due to invalid handle)
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({}, {"get_session_tensor:0"}, {}, &outputs);
        
        // Clean up
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}