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
        
        if (size < 16) return 0;
        
        // Extract dimensions and data format
        uint32_t batch_size = *reinterpret_cast<const uint32_t*>(data + offset) % 10 + 1;
        offset += 4;
        uint32_t height = *reinterpret_cast<const uint32_t*>(data + offset) % 32 + 1;
        offset += 4;
        uint32_t width = *reinterpret_cast<const uint32_t*>(data + offset) % 32 + 1;
        offset += 4;
        uint32_t channels = *reinterpret_cast<const uint32_t*>(data + offset) % 16 + 1;
        offset += 4;
        
        if (offset >= size) return 0;
        
        // Create input tensor shape
        tensorflow::TensorShape input_shape({batch_size, height, width, channels});
        tensorflow::TensorShape bias_shape({channels});
        
        // Calculate required data size
        size_t input_elements = input_shape.num_elements();
        size_t bias_elements = bias_shape.num_elements();
        size_t required_size = (input_elements + bias_elements) * sizeof(float);
        
        if (offset + required_size > size) return 0;
        
        // Create input tensor
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, input_shape);
        auto input_flat = input_tensor.flat<float>();
        
        // Fill input tensor with fuzz data
        const float* input_data = reinterpret_cast<const float*>(data + offset);
        for (int i = 0; i < input_elements && offset + i * sizeof(float) < size; ++i) {
            input_flat(i) = input_data[i];
        }
        offset += input_elements * sizeof(float);
        
        // Create bias tensor
        tensorflow::Tensor bias_tensor(tensorflow::DT_FLOAT, bias_shape);
        auto bias_flat = bias_tensor.flat<float>();
        
        // Fill bias tensor with fuzz data
        const float* bias_data = reinterpret_cast<const float*>(data + offset);
        for (int i = 0; i < bias_elements && offset + i * sizeof(float) < size; ++i) {
            bias_flat(i) = bias_data[i];
        }
        
        // Create session and graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* input_node = graph_def.add_node();
        input_node->set_name("input");
        input_node->set_op("Placeholder");
        (*input_node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        input_node->mutable_attr()->at("dtype").set_type(tensorflow::DT_FLOAT);
        
        tensorflow::NodeDef* bias_node = graph_def.add_node();
        bias_node->set_name("bias");
        bias_node->set_op("Placeholder");
        (*bias_node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        
        tensorflow::NodeDef* bias_add_node = graph_def.add_node();
        bias_add_node->set_name("bias_add");
        bias_add_node->set_op("BiasAdd");
        bias_add_node->add_input("input");
        bias_add_node->add_input("bias");
        (*bias_add_node->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the operation
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"input", input_tensor},
            {"bias", bias_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"bias_add"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Verify output shape
            tensorflow::TensorShape expected_shape = input_shape;
            if (outputs[0].shape() == expected_shape) {
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