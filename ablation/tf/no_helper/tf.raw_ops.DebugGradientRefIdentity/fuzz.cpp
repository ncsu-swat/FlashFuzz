#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/framework/kernel_def_builder.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/status_test_util.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract basic parameters from fuzz input
        int32_t dtype_int = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        int32_t num_dims = (*reinterpret_cast<const int32_t*>(data + offset)) % 4 + 1;
        offset += sizeof(int32_t);
        
        int32_t dim_size = (*reinterpret_cast<const int32_t*>(data + offset)) % 10 + 1;
        offset += sizeof(int32_t);
        
        int32_t num_elements = (*reinterpret_cast<const int32_t*>(data + offset)) % 100 + 1;
        offset += sizeof(int32_t);
        
        // Map to valid TensorFlow data types
        tensorflow::DataType dtype;
        switch (dtype_int % 6) {
            case 0: dtype = tensorflow::DT_FLOAT; break;
            case 1: dtype = tensorflow::DT_DOUBLE; break;
            case 2: dtype = tensorflow::DT_INT32; break;
            case 3: dtype = tensorflow::DT_INT64; break;
            case 4: dtype = tensorflow::DT_BOOL; break;
            default: dtype = tensorflow::DT_STRING; break;
        }
        
        // Create tensor shape
        tensorflow::TensorShape shape;
        for (int i = 0; i < num_dims; ++i) {
            shape.AddDim(dim_size);
        }
        
        // Create input tensor
        tensorflow::Tensor input_tensor(dtype, shape);
        
        // Fill tensor with fuzz data
        size_t tensor_bytes = input_tensor.TotalBytes();
        if (offset + tensor_bytes <= size) {
            std::memcpy(input_tensor.tensor_data().data(), data + offset, tensor_bytes);
        } else {
            // Fill with pattern if not enough data
            auto flat = input_tensor.flat<float>();
            for (int i = 0; i < flat.size(); ++i) {
                flat(i) = static_cast<float>(i % 256);
            }
        }
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Build graph with DebugGradientRefIdentity operation
        tensorflow::GraphDef graph_def;
        tensorflow::GraphDefBuilder builder(tensorflow::GraphDefBuilder::kFailImmediately);
        
        // Create placeholder for input
        auto input_node = tensorflow::ops::Placeholder(builder.opts()
            .WithName("input")
            .WithAttr("dtype", dtype));
        
        // Create DebugGradientRefIdentity node
        tensorflow::NodeDef debug_node;
        debug_node.set_name("debug_gradient_ref_identity");
        debug_node.set_op("DebugGradientRefIdentity");
        debug_node.add_input("input");
        (*debug_node.mutable_attr())["T"].set_type(dtype);
        
        *graph_def.add_node() = input_node.node();
        *graph_def.add_node() = debug_node;
        
        // Create session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0; // Skip if graph creation fails
        }
        
        // Prepare input
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"input", input_tensor}
        };
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"debug_gradient_ref_identity"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Verify output has same shape and type as input
            const tensorflow::Tensor& output = outputs[0];
            if (output.dtype() == input_tensor.dtype() && 
                output.shape().IsSameSize(input_tensor.shape())) {
                // Basic validation passed
            }
        }
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}