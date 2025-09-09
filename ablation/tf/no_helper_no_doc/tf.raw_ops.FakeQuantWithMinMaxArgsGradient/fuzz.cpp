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
        
        if (size < 20) return 0;
        
        // Extract parameters from fuzzer input
        int32_t num_bits = (data[offset] % 16) + 1;
        offset += 1;
        
        bool narrow_range = data[offset] % 2;
        offset += 1;
        
        // Extract dimensions for gradients tensor
        int32_t batch_size = (data[offset] % 8) + 1;
        offset += 1;
        int32_t height = (data[offset] % 8) + 1;
        offset += 1;
        int32_t width = (data[offset] % 8) + 1;
        offset += 1;
        int32_t channels = (data[offset] % 8) + 1;
        offset += 1;
        
        // Extract min and max values
        float min_val = -10.0f + (data[offset] % 200) * 0.1f;
        offset += 1;
        float max_val = min_val + (data[offset] % 100) * 0.1f + 0.1f;
        offset += 1;
        
        // Create TensorFlow session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create graph
        tensorflow::GraphDef graph_def;
        
        // Create gradients input tensor
        tensorflow::TensorShape gradients_shape({batch_size, height, width, channels});
        tensorflow::Tensor gradients_tensor(tensorflow::DT_FLOAT, gradients_shape);
        auto gradients_flat = gradients_tensor.flat<float>();
        
        // Fill gradients tensor with fuzzer data
        for (int i = 0; i < gradients_flat.size() && offset < size; ++i) {
            gradients_flat(i) = -1.0f + (data[offset % size] / 255.0f) * 2.0f;
            offset++;
        }
        
        // Create inputs tensor
        tensorflow::Tensor inputs_tensor(tensorflow::DT_FLOAT, gradients_shape);
        auto inputs_flat = inputs_tensor.flat<float>();
        
        // Fill inputs tensor with fuzzer data
        for (int i = 0; i < inputs_flat.size() && offset < size; ++i) {
            inputs_flat(i) = min_val + (data[offset % size] / 255.0f) * (max_val - min_val);
            offset++;
        }
        
        // Build the node
        auto builder = tensorflow::NodeDefBuilder("fake_quant_grad", "FakeQuantWithMinMaxArgsGradient")
                          .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))
                          .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))
                          .Attr("min", min_val)
                          .Attr("max", max_val)
                          .Attr("num_bits", num_bits)
                          .Attr("narrow_range", narrow_range);
        
        tensorflow::NodeDef node_def;
        auto status = builder.Finalize(&node_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Add node to graph
        auto* node = graph_def.add_node();
        *node = node_def;
        
        // Add placeholder nodes for inputs
        auto* gradients_node = graph_def.add_node();
        gradients_node->set_name("gradients");
        gradients_node->set_op("Placeholder");
        (*gradients_node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        
        auto* inputs_node = graph_def.add_node();
        inputs_node->set_name("inputs");
        inputs_node->set_op("Placeholder");
        (*inputs_node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        
        // Update the main node inputs
        node->clear_input();
        node->add_input("gradients");
        node->add_input("inputs");
        
        // Create session and run
        status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"gradients", gradients_tensor},
            {"inputs", inputs_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {"fake_quant_grad"};
        
        status = session->Run(inputs, output_names, {}, &outputs);
        if (!status.ok()) {
            return 0;
        }
        
        // Verify output
        if (!outputs.empty()) {
            const auto& output = outputs[0];
            if (output.dtype() == tensorflow::DT_FLOAT && 
                output.shape().dims() == gradients_shape.dims()) {
                // Basic validation passed
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}