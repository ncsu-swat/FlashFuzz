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
#include <tensorflow/core/platform/logging.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/graph/node_builder.h>
#include <tensorflow/core/graph/graph.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 8) return 0;
        
        // Extract data type
        uint8_t dtype_val = data[offset++] % 19; // TensorFlow has ~19 common data types
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
        
        // Extract shape dimensions
        uint8_t num_dims = (data[offset++] % 6) + 1; // 1-6 dimensions
        if (offset + num_dims * 4 > size) return 0;
        
        std::vector<int64_t> shape_dims;
        for (int i = 0; i < num_dims && offset + 4 <= size; i++) {
            int32_t dim = *reinterpret_cast<const int32_t*>(data + offset);
            offset += 4;
            // Limit dimension size to avoid memory issues
            dim = std::abs(dim) % 100 + 1;
            shape_dims.push_back(dim);
        }
        
        tensorflow::TensorShape shape(shape_dims);
        
        // Create a graph with placeholder
        tensorflow::Graph graph(tensorflow::OpRegistry::Global());
        
        tensorflow::Node* placeholder_node;
        tensorflow::NodeBuilder builder("test_placeholder", "Placeholder");
        builder.Attr("dtype", dtype);
        builder.Attr("shape", shape);
        
        tensorflow::Status status = builder.Finalize(&graph, &placeholder_node);
        if (!status.ok()) {
            return 0;
        }
        
        // Convert graph to GraphDef
        tensorflow::GraphDef graph_def;
        graph.ToGraphDef(&graph_def);
        
        // Create session and add the graph
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Create a tensor to feed the placeholder
        tensorflow::Tensor input_tensor(dtype, shape);
        
        // Fill tensor with some data based on remaining fuzz input
        if (dtype == tensorflow::DT_FLOAT && offset < size) {
            auto flat = input_tensor.flat<float>();
            for (int i = 0; i < flat.size() && offset < size; i++) {
                flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
                offset++;
            }
        } else if (dtype == tensorflow::DT_INT32 && offset < size) {
            auto flat = input_tensor.flat<int32_t>();
            for (int i = 0; i < flat.size() && offset + 4 <= size; i++) {
                flat(i) = *reinterpret_cast<const int32_t*>(data + offset);
                offset += 4;
            }
        } else if (dtype == tensorflow::DT_BOOL && offset < size) {
            auto flat = input_tensor.flat<bool>();
            for (int i = 0; i < flat.size() && offset < size; i++) {
                flat(i) = (data[offset] % 2) == 1;
                offset++;
            }
        }
        
        // Run the session with the placeholder
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({{"test_placeholder", input_tensor}}, {"test_placeholder"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Successfully created and used placeholder
        }
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}