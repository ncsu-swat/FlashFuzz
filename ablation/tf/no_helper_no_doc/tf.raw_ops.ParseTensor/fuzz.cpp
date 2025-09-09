#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/str_util.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/kernels/ops_util.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 8) return 0;
        
        // Extract data type from fuzzer input
        uint8_t dtype_val = data[offset++] % 19; // TensorFlow has ~19 basic data types
        tensorflow::DataType out_type;
        switch (dtype_val) {
            case 0: out_type = tensorflow::DT_FLOAT; break;
            case 1: out_type = tensorflow::DT_DOUBLE; break;
            case 2: out_type = tensorflow::DT_INT32; break;
            case 3: out_type = tensorflow::DT_UINT8; break;
            case 4: out_type = tensorflow::DT_INT16; break;
            case 5: out_type = tensorflow::DT_INT8; break;
            case 6: out_type = tensorflow::DT_STRING; break;
            case 7: out_type = tensorflow::DT_COMPLEX64; break;
            case 8: out_type = tensorflow::DT_INT64; break;
            case 9: out_type = tensorflow::DT_BOOL; break;
            case 10: out_type = tensorflow::DT_QINT8; break;
            case 11: out_type = tensorflow::DT_QUINT8; break;
            case 12: out_type = tensorflow::DT_QINT32; break;
            case 13: out_type = tensorflow::DT_BFLOAT16; break;
            case 14: out_type = tensorflow::DT_QINT16; break;
            case 15: out_type = tensorflow::DT_QUINT16; break;
            case 16: out_type = tensorflow::DT_UINT16; break;
            case 17: out_type = tensorflow::DT_COMPLEX128; break;
            default: out_type = tensorflow::DT_HALF; break;
        }
        
        // Create serialized tensor data from remaining fuzzer input
        std::string serialized_tensor;
        if (offset < size) {
            serialized_tensor = std::string(reinterpret_cast<const char*>(data + offset), size - offset);
        }
        
        // Create input tensor for serialized data
        tensorflow::Tensor serialized_input(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        serialized_input.scalar<tensorflow::tstring>()() = serialized_tensor;
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Build graph with ParseTensor operation
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* parse_node = graph_def.add_node();
        parse_node->set_name("parse_tensor");
        parse_node->set_op("ParseTensor");
        parse_node->add_input("serialized_tensor:0");
        
        // Set attributes
        tensorflow::AttrValue dtype_attr;
        dtype_attr.set_type(out_type);
        (*parse_node->mutable_attr())["out_type"] = dtype_attr;
        
        // Add placeholder for input
        tensorflow::NodeDef* input_node = graph_def.add_node();
        input_node->set_name("serialized_tensor");
        input_node->set_op("Placeholder");
        tensorflow::AttrValue input_dtype_attr;
        input_dtype_attr.set_type(tensorflow::DT_STRING);
        (*input_node->mutable_attr())["dtype"] = input_dtype_attr;
        
        // Create the session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"serialized_tensor:0", serialized_input}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {"parse_tensor:0"};
        
        status = session->Run(inputs, output_names, {}, &outputs);
        
        // Clean up
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}