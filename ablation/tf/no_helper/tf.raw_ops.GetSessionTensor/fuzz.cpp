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
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/platform/logging.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 8) return 0;
        
        // Extract handle string length
        uint32_t handle_len = 0;
        if (offset + sizeof(handle_len) > size) return 0;
        memcpy(&handle_len, data + offset, sizeof(handle_len));
        offset += sizeof(handle_len);
        
        // Limit handle length to prevent excessive memory usage
        handle_len = handle_len % 1024;
        if (handle_len == 0) handle_len = 1;
        
        if (offset + handle_len > size) return 0;
        
        // Extract handle string
        std::string handle_str(reinterpret_cast<const char*>(data + offset), handle_len);
        offset += handle_len;
        
        // Extract dtype
        if (offset + sizeof(uint32_t) > size) return 0;
        uint32_t dtype_val = 0;
        memcpy(&dtype_val, data + offset, sizeof(dtype_val));
        offset += sizeof(dtype_val);
        
        // Map to valid TensorFlow data types
        tensorflow::DataType dtype;
        switch (dtype_val % 10) {
            case 0: dtype = tensorflow::DT_FLOAT; break;
            case 1: dtype = tensorflow::DT_DOUBLE; break;
            case 2: dtype = tensorflow::DT_INT32; break;
            case 3: dtype = tensorflow::DT_INT64; break;
            case 4: dtype = tensorflow::DT_UINT8; break;
            case 5: dtype = tensorflow::DT_STRING; break;
            case 6: dtype = tensorflow::DT_BOOL; break;
            case 7: dtype = tensorflow::DT_INT16; break;
            case 8: dtype = tensorflow::DT_INT8; break;
            default: dtype = tensorflow::DT_FLOAT; break;
        }
        
        // Create a simple graph with GetSessionTensor operation
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node = graph_def.add_node();
        node->set_name("get_session_tensor");
        node->set_op("GetSessionTensor");
        
        // Add handle input
        tensorflow::NodeDef* handle_node = graph_def.add_node();
        handle_node->set_name("handle_input");
        handle_node->set_op("Const");
        auto handle_attr = handle_node->mutable_attr();
        (*handle_attr)["dtype"].set_type(tensorflow::DT_STRING);
        
        tensorflow::TensorProto handle_tensor_proto;
        handle_tensor_proto.set_dtype(tensorflow::DT_STRING);
        handle_tensor_proto.add_string_val(handle_str);
        (*handle_attr)["value"].mutable_tensor()->CopyFrom(handle_tensor_proto);
        
        node->add_input("handle_input");
        
        // Set dtype attribute
        auto attr = node->mutable_attr();
        (*attr)["dtype"].set_type(dtype);
        
        // Create session and try to run
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        if (session) {
            tensorflow::Status status = session->Create(graph_def);
            if (status.ok()) {
                std::vector<tensorflow::Tensor> outputs;
                status = session->Run({}, {"get_session_tensor:0"}, {}, &outputs);
                // We expect this to fail since we don't have a valid session tensor handle
                // but we're testing that the operation doesn't crash
            }
            session->Close();
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}