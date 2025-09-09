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
        
        // Extract parameters from fuzzer input
        int32_t num_bits = (data[offset] % 16) + 1; // 1-16 bits
        offset += 1;
        
        bool signed_input = data[offset] % 2;
        offset += 1;
        
        bool range_given = data[offset] % 2;
        offset += 1;
        
        bool narrow_range = data[offset] % 2;
        offset += 1;
        
        int32_t axis = static_cast<int32_t>(data[offset] % 4) - 2; // -2 to 1
        offset += 1;
        
        // Extract tensor dimensions
        if (offset + 8 > size) return 0;
        int32_t dim1 = (data[offset] % 8) + 1;
        int32_t dim2 = (data[offset + 1] % 8) + 1;
        offset += 2;
        
        // Calculate required data size
        size_t tensor_size = dim1 * dim2;
        size_t float_data_size = tensor_size * sizeof(float);
        
        if (offset + float_data_size + 2 * sizeof(float) > size) return 0;
        
        // Create input tensor
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({dim1, dim2}));
        auto input_flat = input_tensor.flat<float>();
        
        // Fill tensor with fuzzer data
        const float* float_data = reinterpret_cast<const float*>(data + offset);
        for (int i = 0; i < tensor_size; i++) {
            input_flat(i) = float_data[i];
        }
        offset += float_data_size;
        
        // Extract input_min and input_max
        float input_min = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        float input_max = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        
        // Ensure input_min < input_max
        if (input_min >= input_max) {
            input_max = input_min + 1.0f;
        }
        
        tensorflow::Tensor input_min_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        input_min_tensor.scalar<float>()() = input_min;
        
        tensorflow::Tensor input_max_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        input_max_tensor.scalar<float>()() = input_max;
        
        // Create session and graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        
        node_def->set_name("quantize_and_dequantize_v3");
        node_def->set_op("QuantizeAndDequantizeV3");
        node_def->add_input("input:0");
        node_def->add_input("input_min:0");
        node_def->add_input("input_max:0");
        
        // Set attributes
        tensorflow::AttrValue num_bits_attr;
        num_bits_attr.set_i(num_bits);
        (*node_def->mutable_attr())["num_bits"] = num_bits_attr;
        
        tensorflow::AttrValue signed_input_attr;
        signed_input_attr.set_b(signed_input);
        (*node_def->mutable_attr())["signed_input"] = signed_input_attr;
        
        tensorflow::AttrValue range_given_attr;
        range_given_attr.set_b(range_given);
        (*node_def->mutable_attr())["range_given"] = range_given_attr;
        
        tensorflow::AttrValue narrow_range_attr;
        narrow_range_attr.set_b(narrow_range);
        (*node_def->mutable_attr())["narrow_range"] = narrow_range_attr;
        
        tensorflow::AttrValue axis_attr;
        axis_attr.set_i(axis);
        (*node_def->mutable_attr())["axis"] = axis_attr;
        
        // Add input nodes
        tensorflow::NodeDef* input_node = graph_def.add_node();
        input_node->set_name("input");
        input_node->set_op("Placeholder");
        tensorflow::AttrValue dtype_attr;
        dtype_attr.set_type(tensorflow::DT_FLOAT);
        (*input_node->mutable_attr())["dtype"] = dtype_attr;
        
        tensorflow::NodeDef* input_min_node = graph_def.add_node();
        input_min_node->set_name("input_min");
        input_min_node->set_op("Placeholder");
        (*input_min_node->mutable_attr())["dtype"] = dtype_attr;
        
        tensorflow::NodeDef* input_max_node = graph_def.add_node();
        input_max_node->set_name("input_max");
        input_max_node->set_op("Placeholder");
        (*input_max_node->mutable_attr())["dtype"] = dtype_attr;
        
        // Create session
        tensorflow::SessionOptions session_options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(session_options));
        
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Prepare inputs
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"input:0", input_tensor},
            {"input_min:0", input_min_tensor},
            {"input_max:0", input_max_tensor}
        };
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"quantize_and_dequantize_v3:0"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Verify output shape matches input shape
            if (outputs[0].shape() == input_tensor.shape()) {
                // Access output data to ensure it's computed
                auto output_flat = outputs[0].flat<float>();
                volatile float sum = 0.0f;
                for (int i = 0; i < output_flat.size(); i++) {
                    sum += output_flat(i);
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