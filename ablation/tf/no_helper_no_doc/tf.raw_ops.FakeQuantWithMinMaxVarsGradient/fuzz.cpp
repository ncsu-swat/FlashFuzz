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
        
        // Extract dimensions and parameters from fuzz input
        int batch_size = (data[offset] % 8) + 1;
        offset++;
        int height = (data[offset] % 8) + 1;
        offset++;
        int width = (data[offset] % 8) + 1;
        offset++;
        int channels = (data[offset] % 8) + 1;
        offset++;
        
        int num_bits = (data[offset] % 8) + 1;
        offset++;
        bool narrow_range = data[offset] % 2;
        offset++;
        
        if (offset + (batch_size * height * width * channels + 2) * sizeof(float) > size) {
            return 0;
        }
        
        // Create input tensors
        tensorflow::Tensor gradients(tensorflow::DT_FLOAT, 
                                   tensorflow::TensorShape({batch_size, height, width, channels}));
        tensorflow::Tensor inputs(tensorflow::DT_FLOAT, 
                                tensorflow::TensorShape({batch_size, height, width, channels}));
        tensorflow::Tensor min_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        tensorflow::Tensor max_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        
        // Fill tensors with fuzz data
        auto gradients_flat = gradients.flat<float>();
        auto inputs_flat = inputs.flat<float>();
        
        for (int i = 0; i < gradients_flat.size() && offset + sizeof(float) <= size; i++) {
            float val;
            memcpy(&val, data + offset, sizeof(float));
            gradients_flat(i) = val;
            offset += sizeof(float);
        }
        
        for (int i = 0; i < inputs_flat.size() && offset + sizeof(float) <= size; i++) {
            float val;
            memcpy(&val, data + offset, sizeof(float));
            inputs_flat(i) = val;
            offset += sizeof(float);
        }
        
        if (offset + 2 * sizeof(float) <= size) {
            float min_val, max_val;
            memcpy(&min_val, data + offset, sizeof(float));
            offset += sizeof(float);
            memcpy(&max_val, data + offset, sizeof(float));
            offset += sizeof(float);
            
            min_tensor.scalar<float>()() = min_val;
            max_tensor.scalar<float>()() = max_val;
        } else {
            min_tensor.scalar<float>()() = -1.0f;
            max_tensor.scalar<float>()() = 1.0f;
        }
        
        // Create session and graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        
        node_def->set_name("fake_quant_grad");
        node_def->set_op("FakeQuantWithMinMaxVarsGradient");
        node_def->add_input("gradients:0");
        node_def->add_input("inputs:0");
        node_def->add_input("min:0");
        node_def->add_input("max:0");
        
        // Set attributes
        tensorflow::AttrValue num_bits_attr;
        num_bits_attr.set_i(num_bits);
        (*node_def->mutable_attr())["num_bits"] = num_bits_attr;
        
        tensorflow::AttrValue narrow_range_attr;
        narrow_range_attr.set_b(narrow_range);
        (*node_def->mutable_attr())["narrow_range"] = narrow_range_attr;
        
        // Add placeholder nodes
        tensorflow::NodeDef* gradients_node = graph_def.add_node();
        gradients_node->set_name("gradients");
        gradients_node->set_op("Placeholder");
        tensorflow::AttrValue dtype_attr;
        dtype_attr.set_type(tensorflow::DT_FLOAT);
        (*gradients_node->mutable_attr())["dtype"] = dtype_attr;
        
        tensorflow::NodeDef* inputs_node = graph_def.add_node();
        inputs_node->set_name("inputs");
        inputs_node->set_op("Placeholder");
        (*inputs_node->mutable_attr())["dtype"] = dtype_attr;
        
        tensorflow::NodeDef* min_node = graph_def.add_node();
        min_node->set_name("min");
        min_node->set_op("Placeholder");
        (*min_node->mutable_attr())["dtype"] = dtype_attr;
        
        tensorflow::NodeDef* max_node = graph_def.add_node();
        max_node->set_name("max");
        max_node->set_op("Placeholder");
        (*max_node->mutable_attr())["dtype"] = dtype_attr;
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Prepare feed dict
        std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict = {
            {"gradients:0", gradients},
            {"inputs:0", inputs},
            {"min:0", min_tensor},
            {"max:0", max_tensor}
        };
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {
            "fake_quant_grad:0",
            "fake_quant_grad:1", 
            "fake_quant_grad:2"
        };
        
        status = session->Run(feed_dict, output_names, {}, &outputs);
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}