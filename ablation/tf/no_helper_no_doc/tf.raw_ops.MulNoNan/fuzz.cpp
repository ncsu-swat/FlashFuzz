#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/kernel_def_builder.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/framework/node_def_util.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions for tensors
        uint32_t dim1 = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        uint32_t dim2 = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        
        // Limit dimensions to reasonable values
        dim1 = (dim1 % 10) + 1;
        dim2 = (dim2 % 10) + 1;
        
        size_t tensor_size = dim1 * dim2;
        size_t float_bytes = tensor_size * sizeof(float);
        
        if (offset + 2 * float_bytes > size) return 0;
        
        // Create TensorFlow session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create graph definition
        tensorflow::GraphDef graph_def;
        
        // Add placeholder nodes for inputs
        tensorflow::NodeDef* x_node = graph_def.add_node();
        x_node->set_name("x");
        x_node->set_op("Placeholder");
        tensorflow::AddNodeAttr("dtype", tensorflow::DT_FLOAT, x_node);
        tensorflow::AddNodeAttr("shape", tensorflow::TensorShape({static_cast<int64_t>(dim1), static_cast<int64_t>(dim2)}), x_node);
        
        tensorflow::NodeDef* y_node = graph_def.add_node();
        y_node->set_name("y");
        y_node->set_op("Placeholder");
        tensorflow::AddNodeAttr("dtype", tensorflow::DT_FLOAT, y_node);
        tensorflow::AddNodeAttr("shape", tensorflow::TensorShape({static_cast<int64_t>(dim1), static_cast<int64_t>(dim2)}), y_node);
        
        // Add MulNoNan operation
        tensorflow::NodeDef* mul_node = graph_def.add_node();
        mul_node->set_name("mul_no_nan");
        mul_node->set_op("MulNoNan");
        mul_node->add_input("x");
        mul_node->add_input("y");
        tensorflow::AddNodeAttr("T", tensorflow::DT_FLOAT, mul_node);
        
        // Create session with graph
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Create input tensors
        tensorflow::Tensor x_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({static_cast<int64_t>(dim1), static_cast<int64_t>(dim2)}));
        tensorflow::Tensor y_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({static_cast<int64_t>(dim1), static_cast<int64_t>(dim2)}));
        
        // Fill tensors with fuzz data
        auto x_flat = x_tensor.flat<float>();
        auto y_flat = y_tensor.flat<float>();
        
        const float* x_data = reinterpret_cast<const float*>(data + offset);
        const float* y_data = reinterpret_cast<const float*>(data + offset + float_bytes);
        
        for (size_t i = 0; i < tensor_size; ++i) {
            x_flat(i) = x_data[i];
            y_flat(i) = y_data[i];
        }
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({{"x", x_tensor}, {"y", y_tensor}}, {"mul_no_nan"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Verify output tensor properties
            const tensorflow::Tensor& result = outputs[0];
            if (result.dtype() == tensorflow::DT_FLOAT && 
                result.shape().dims() == 2 &&
                result.shape().dim_size(0) == dim1 &&
                result.shape().dim_size(1) == dim2) {
                
                // Access result data to ensure computation completed
                auto result_flat = result.flat<float>();
                volatile float sum = 0.0f;
                for (int i = 0; i < result_flat.size(); ++i) {
                    sum += result_flat(i);
                }
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