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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < sizeof(int32_t) * 2) {
            return 0;
        }
        
        // Extract tensor dimensions
        int32_t num_dims = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        // Limit dimensions to reasonable range
        num_dims = std::max(1, std::min(num_dims, 4));
        
        if (offset + num_dims * sizeof(int32_t) > size) {
            return 0;
        }
        
        // Extract shape dimensions
        std::vector<int64_t> dims;
        int64_t total_elements = 1;
        for (int i = 0; i < num_dims; i++) {
            int32_t dim = *reinterpret_cast<const int32_t*>(data + offset);
            offset += sizeof(int32_t);
            dim = std::max(1, std::min(dim, 100)); // Limit dimension size
            dims.push_back(dim);
            total_elements *= dim;
        }
        
        // Limit total elements to prevent excessive memory usage
        if (total_elements > 10000) {
            return 0;
        }
        
        tensorflow::TensorShape shape(dims);
        
        // Calculate required data size for float values
        size_t required_data_size = total_elements * sizeof(float);
        if (offset + required_data_size > size) {
            return 0;
        }
        
        // Create input tensor
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, shape);
        auto input_flat = input_tensor.flat<float>();
        
        // Fill tensor with fuzz data
        const float* float_data = reinterpret_cast<const float*>(data + offset);
        for (int64_t i = 0; i < total_elements; i++) {
            float val = float_data[i];
            // Handle NaN and infinity values
            if (std::isnan(val) || std::isinf(val)) {
                val = 0.0f;
            }
            input_flat(i) = val;
        }
        
        // Create a simple computation graph with Sin operation
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* input_node = graph_def.add_node();
        input_node->set_name("input");
        input_node->set_op("Placeholder");
        (*input_node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        (*input_node->mutable_attr())["shape"].mutable_shape();
        
        tensorflow::NodeDef* sin_node = graph_def.add_node();
        sin_node->set_name("sin_output");
        sin_node->set_op("Sin");
        sin_node->add_input("input");
        (*sin_node->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
        
        // Create session and run the operation
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"input", input_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"sin_output"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Verify output tensor has expected properties
            const tensorflow::Tensor& output = outputs[0];
            if (output.dtype() == tensorflow::DT_FLOAT && 
                output.shape().IsSameSize(shape)) {
                auto output_flat = output.flat<float>();
                // Basic sanity check on output values
                for (int64_t i = 0; i < total_elements; i++) {
                    float result = output_flat(i);
                    // Sin output should be in range [-1, 1] for finite inputs
                    if (std::isfinite(result) && (result < -1.1f || result > 1.1f)) {
                        break;
                    }
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