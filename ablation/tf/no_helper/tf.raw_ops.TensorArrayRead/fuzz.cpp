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
        
        if (size < 16) return 0;
        
        // Extract handle string length and content
        uint32_t handle_len = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += sizeof(uint32_t);
        handle_len = handle_len % 1024; // Limit size
        
        if (offset + handle_len >= size) return 0;
        
        std::string handle_str(reinterpret_cast<const char*>(data + offset), handle_len);
        offset += handle_len;
        
        // Extract index value
        if (offset + sizeof(int32_t) >= size) return 0;
        int32_t index_val = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        // Extract flow_in value
        if (offset + sizeof(float) >= size) return 0;
        float flow_val = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        
        // Extract dtype
        if (offset + sizeof(uint8_t) >= size) return 0;
        uint8_t dtype_val = data[offset] % 19; // Limit to valid TF dtypes
        offset += sizeof(uint8_t);
        
        tensorflow::DataType dtype;
        switch (dtype_val) {
            case 0: dtype = tensorflow::DT_FLOAT; break;
            case 1: dtype = tensorflow::DT_DOUBLE; break;
            case 2: dtype = tensorflow::DT_INT32; break;
            case 3: dtype = tensorflow::DT_UINT8; break;
            case 4: dtype = tensorflow::DT_INT16; break;
            case 5: dtype = tensorflow::DT_INT8; break;
            case 6: dtype = tensorflow::DT_STRING; break;
            case 7: dtype = tensorflow::DT_COMPLEX64; break;
            case 8: dtype = tensorflow::DT_INT64; break;
            case 9: dtype = tensorflow::DT_BOOL; break;
            case 10: dtype = tensorflow::DT_QINT8; break;
            case 11: dtype = tensorflow::DT_QUINT8; break;
            case 12: dtype = tensorflow::DT_QINT32; break;
            case 13: dtype = tensorflow::DT_BFLOAT16; break;
            case 14: dtype = tensorflow::DT_QINT16; break;
            case 15: dtype = tensorflow::DT_QUINT16; break;
            case 16: dtype = tensorflow::DT_UINT16; break;
            case 17: dtype = tensorflow::DT_COMPLEX128; break;
            case 18: dtype = tensorflow::DT_HALF; break;
            default: dtype = tensorflow::DT_FLOAT; break;
        }
        
        // Create tensors
        tensorflow::Tensor handle_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        handle_tensor.scalar<tensorflow::tstring>()() = tensorflow::tstring(handle_str);
        
        tensorflow::Tensor index_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        index_tensor.scalar<int32_t>()() = index_val;
        
        tensorflow::Tensor flow_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        flow_tensor.scalar<float>()() = flow_val;
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create graph
        tensorflow::GraphDef graph_def;
        tensorflow::GraphDefBuilder builder(tensorflow::GraphDefBuilder::kFailImmediately);
        
        auto handle_node = tensorflow::ops::Const(handle_tensor, builder.opts().WithName("handle"));
        auto index_node = tensorflow::ops::Const(index_tensor, builder.opts().WithName("index"));
        auto flow_node = tensorflow::ops::Const(flow_tensor, builder.opts().WithName("flow_in"));
        
        // Create TensorArrayRead operation
        auto read_node = tensorflow::ops::UnaryOp("TensorArrayReadV3", handle_node,
                                                 builder.opts()
                                                 .WithName("tensor_array_read")
                                                 .WithAttr("dtype", dtype));
        
        tensorflow::Status status = builder.ToGraphDef(&graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({{"handle:0", handle_tensor}, 
                              {"index:0", index_tensor}, 
                              {"flow_in:0", flow_tensor}}, 
                             {"tensor_array_read:0"}, {}, &outputs);
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}