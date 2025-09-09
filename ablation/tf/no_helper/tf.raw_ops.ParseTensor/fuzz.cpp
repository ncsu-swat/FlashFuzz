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
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/status_test_util.h>
#include <tensorflow/core/framework/tensor.pb.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < sizeof(uint32_t)) {
            return 0;
        }
        
        // Extract data type from first 4 bytes
        uint32_t dtype_val;
        memcpy(&dtype_val, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        // Map to valid TensorFlow data types
        tensorflow::DataType out_type;
        switch (dtype_val % 19) {
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
            case 18: out_type = tensorflow::DT_HALF; break;
            default: out_type = tensorflow::DT_FLOAT; break;
        }
        
        // Use remaining data as serialized tensor proto
        std::string serialized_data(reinterpret_cast<const char*>(data + offset), size - offset);
        
        // Create a session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Build graph with ParseTensor operation
        tensorflow::GraphDef graph_def;
        tensorflow::GraphDefBuilder builder(tensorflow::GraphDefBuilder::kFailImmediately);
        
        // Create input placeholder for serialized data
        auto serialized_input = tensorflow::ops::Placeholder(
            builder.opts().WithName("serialized_input").WithAttr("dtype", tensorflow::DT_STRING));
        
        // Create ParseTensor operation
        auto parse_tensor_op = tensorflow::ops::UnaryOp(
            "ParseTensor", serialized_input,
            builder.opts().WithName("parse_tensor").WithAttr("out_type", out_type));
        
        tensorflow::Status status = builder.ToGraphDef(&graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Create input tensor
        tensorflow::Tensor input_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        input_tensor.scalar<tensorflow::tstring>()() = serialized_data;
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({{"serialized_input", input_tensor}}, {"parse_tensor"}, {}, &outputs);
        
        // Clean up
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}