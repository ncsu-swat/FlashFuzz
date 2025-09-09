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
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/const_op.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions and data type
        uint32_t x_dim = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        uint32_t y_dim = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        uint32_t data_type = *reinterpret_cast<const uint32_t*>(data + offset) % 6;
        offset += 4;
        uint32_t num_elements = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        
        // Limit dimensions and elements to reasonable values
        x_dim = std::min(x_dim % 10 + 1, 5u);
        y_dim = std::min(y_dim % 10 + 1, 5u);
        num_elements = std::min(num_elements % 100 + 1, 50u);
        
        tensorflow::DataType dtype;
        size_t element_size;
        
        switch (data_type) {
            case 0:
                dtype = tensorflow::DT_FLOAT;
                element_size = sizeof(float);
                break;
            case 1:
                dtype = tensorflow::DT_DOUBLE;
                element_size = sizeof(double);
                break;
            case 2:
                dtype = tensorflow::DT_HALF;
                element_size = sizeof(tensorflow::bfloat16);
                break;
            case 3:
                dtype = tensorflow::DT_BFLOAT16;
                element_size = sizeof(tensorflow::bfloat16);
                break;
            case 4:
                dtype = tensorflow::DT_COMPLEX64;
                element_size = sizeof(tensorflow::complex64);
                break;
            default:
                dtype = tensorflow::DT_COMPLEX128;
                element_size = sizeof(tensorflow::complex128);
                break;
        }
        
        size_t required_size = num_elements * element_size * 2; // for both x and y
        if (offset + required_size > size) return 0;
        
        // Create tensor shapes
        tensorflow::TensorShape x_shape({static_cast<int64_t>(x_dim), static_cast<int64_t>(num_elements / x_dim)});
        tensorflow::TensorShape y_shape({static_cast<int64_t>(y_dim), static_cast<int64_t>(num_elements / y_dim)});
        
        // Create tensors
        tensorflow::Tensor x_tensor(dtype, x_shape);
        tensorflow::Tensor y_tensor(dtype, y_shape);
        
        // Fill tensors with fuzz data
        if (dtype == tensorflow::DT_FLOAT) {
            auto x_flat = x_tensor.flat<float>();
            auto y_flat = y_tensor.flat<float>();
            
            for (int i = 0; i < x_flat.size() && offset + sizeof(float) <= size; ++i) {
                x_flat(i) = *reinterpret_cast<const float*>(data + offset);
                offset += sizeof(float);
            }
            for (int i = 0; i < y_flat.size() && offset + sizeof(float) <= size; ++i) {
                y_flat(i) = *reinterpret_cast<const float*>(data + offset);
                offset += sizeof(float);
            }
        } else if (dtype == tensorflow::DT_DOUBLE) {
            auto x_flat = x_tensor.flat<double>();
            auto y_flat = y_tensor.flat<double>();
            
            for (int i = 0; i < x_flat.size() && offset + sizeof(double) <= size; ++i) {
                x_flat(i) = *reinterpret_cast<const double*>(data + offset);
                offset += sizeof(double);
            }
            for (int i = 0; i < y_flat.size() && offset + sizeof(double) <= size; ++i) {
                y_flat(i) = *reinterpret_cast<const double*>(data + offset);
                offset += sizeof(double);
            }
        }
        
        // Create a simple graph with MulNoNan operation
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        auto x_placeholder = tensorflow::ops::Placeholder(root.WithOpName("x"), dtype);
        auto y_placeholder = tensorflow::ops::Placeholder(root.WithOpName("y"), dtype);
        
        // Create MulNoNan operation
        tensorflow::Node* mul_no_nan_node;
        tensorflow::NodeBuilder("mul_no_nan", "MulNoNan")
            .Input(x_placeholder.node())
            .Input(y_placeholder.node())
            .Finalize(root.graph(), &mul_no_nan_node);
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::ClientSession::FeedType feeds = {
            {x_placeholder, x_tensor},
            {y_placeholder, y_tensor}
        };
        
        auto status = session.Run(feeds, {tensorflow::Output(mul_no_nan_node)}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Operation completed successfully
            return 0;
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}