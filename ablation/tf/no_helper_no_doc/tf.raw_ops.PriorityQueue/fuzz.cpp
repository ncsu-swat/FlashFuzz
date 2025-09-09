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
        
        if (size < 16) {
            return 0;
        }
        
        // Extract parameters from fuzzer input
        int32_t capacity = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        // Ensure capacity is reasonable to avoid excessive memory usage
        capacity = std::abs(capacity) % 1000 + 1;
        
        // Extract component types count
        uint8_t num_types = data[offset] % 10 + 1; // Limit to reasonable number
        offset += 1;
        
        if (offset + num_types * sizeof(uint8_t) > size) {
            return 0;
        }
        
        // Build component types vector
        std::vector<tensorflow::DataType> component_types;
        for (int i = 0; i < num_types && offset < size; ++i) {
            uint8_t type_val = data[offset] % 19; // TensorFlow has ~19 basic data types
            tensorflow::DataType dt;
            switch (type_val) {
                case 0: dt = tensorflow::DT_FLOAT; break;
                case 1: dt = tensorflow::DT_DOUBLE; break;
                case 2: dt = tensorflow::DT_INT32; break;
                case 3: dt = tensorflow::DT_UINT8; break;
                case 4: dt = tensorflow::DT_INT16; break;
                case 5: dt = tensorflow::DT_INT8; break;
                case 6: dt = tensorflow::DT_STRING; break;
                case 7: dt = tensorflow::DT_COMPLEX64; break;
                case 8: dt = tensorflow::DT_INT64; break;
                case 9: dt = tensorflow::DT_BOOL; break;
                case 10: dt = tensorflow::DT_QINT8; break;
                case 11: dt = tensorflow::DT_QUINT8; break;
                case 12: dt = tensorflow::DT_QINT32; break;
                case 13: dt = tensorflow::DT_BFLOAT16; break;
                case 14: dt = tensorflow::DT_QINT16; break;
                case 15: dt = tensorflow::DT_QUINT16; break;
                case 16: dt = tensorflow::DT_UINT16; break;
                case 17: dt = tensorflow::DT_COMPLEX128; break;
                case 18: dt = tensorflow::DT_HALF; break;
                default: dt = tensorflow::DT_FLOAT; break;
            }
            component_types.push_back(dt);
            offset += 1;
        }
        
        // Extract shapes count
        if (offset >= size) return 0;
        uint8_t num_shapes = data[offset] % num_types + 1;
        offset += 1;
        
        std::vector<tensorflow::TensorShape> shapes;
        for (int i = 0; i < num_shapes && offset + 4 <= size; ++i) {
            // Extract number of dimensions
            uint8_t num_dims = data[offset] % 5; // Limit dimensions
            offset += 1;
            
            std::vector<int64_t> dims;
            for (int j = 0; j < num_dims && offset + 2 <= size; ++j) {
                int16_t dim = *reinterpret_cast<const int16_t*>(data + offset);
                dims.push_back(std::abs(dim) % 100 + 1); // Reasonable dimension size
                offset += 2;
            }
            
            if (dims.empty()) {
                shapes.push_back(tensorflow::TensorShape({})); // Scalar
            } else {
                shapes.push_back(tensorflow::TensorShape(dims));
            }
        }
        
        // Pad shapes vector to match component_types size
        while (shapes.size() < component_types.size()) {
            shapes.push_back(tensorflow::TensorShape({}));
        }
        
        // Extract container and shared_name
        std::string container = "";
        std::string shared_name = "";
        
        if (offset + 2 <= size) {
            uint8_t container_len = data[offset] % 20;
            offset += 1;
            if (offset + container_len <= size) {
                container = std::string(reinterpret_cast<const char*>(data + offset), container_len);
                offset += container_len;
            }
        }
        
        if (offset + 2 <= size) {
            uint8_t shared_name_len = data[offset] % 20;
            offset += 1;
            if (offset + shared_name_len <= size) {
                shared_name = std::string(reinterpret_cast<const char*>(data + offset), shared_name_len);
                offset += shared_name_len;
            }
        }
        
        // Create TensorFlow session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create GraphDef
        tensorflow::GraphDef graph_def;
        tensorflow::GraphDefBuilder builder(tensorflow::GraphDefBuilder::kFailImmediately);
        
        // Create PriorityQueue node
        auto priority_queue_node = tensorflow::ops::SourceOp("PriorityQueue", 
            builder.opts()
                .WithName("priority_queue")
                .WithAttr("capacity", capacity)
                .WithAttr("component_types", component_types)
                .WithAttr("shapes", shapes)
                .WithAttr("container", container)
                .WithAttr("shared_name", shared_name));
        
        tensorflow::Status status = builder.ToGraphDef(&graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Create and run session
        status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({}, {"priority_queue:0"}, {}, &outputs);
        
        // Clean up
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}