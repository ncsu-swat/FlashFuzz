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
        
        if (size < 8) return 0;
        
        // Extract dimensions for input tensor
        uint32_t num_elements = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        
        uint32_t string_length = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        
        // Limit sizes to prevent excessive memory usage
        num_elements = num_elements % 1000 + 1;
        string_length = string_length % 100 + 1;
        
        if (offset + string_length > size) return 0;
        
        // Create input tensor with string data
        tensorflow::Tensor input_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({static_cast<int64_t>(num_elements)}));
        auto input_flat = input_tensor.flat<tensorflow::tstring>();
        
        // Fill tensor with fuzz data
        for (int i = 0; i < num_elements; ++i) {
            size_t start_pos = offset + (i * string_length / num_elements) % (size - offset);
            size_t actual_length = std::min(string_length, size - start_pos);
            if (actual_length > 0) {
                input_flat(i) = tensorflow::tstring(reinterpret_cast<const char*>(data + start_pos), actual_length);
            } else {
                input_flat(i) = tensorflow::tstring("");
            }
        }
        
        // Create session and graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* fingerprint_node = graph_def.add_node();
        fingerprint_node->set_name("fingerprint");
        fingerprint_node->set_op("Fingerprint");
        fingerprint_node->add_input("input:0");
        
        tensorflow::NodeDef* input_node = graph_def.add_node();
        input_node->set_name("input");
        input_node->set_op("Placeholder");
        (*input_node->mutable_attr())["dtype"].set_type(tensorflow::DT_STRING);
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        if (!session) return 0;
        
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) return 0;
        
        // Run the operation
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {{"input:0", input_tensor}};
        std::vector<tensorflow::Tensor> outputs;
        
        status = session->Run(inputs, {"fingerprint:0"}, {}, &outputs);
        if (status.ok() && !outputs.empty()) {
            // Successfully executed fingerprint operation
            auto output_flat = outputs[0].flat<uint8>();
            // The output should be a fingerprint (typically 16 bytes)
            if (output_flat.size() > 0) {
                // Verify output is reasonable
                for (int i = 0; i < std::min(16, static_cast<int>(output_flat.size())); ++i) {
                    volatile uint8_t val = output_flat(i);
                    (void)val; // Use the value to prevent optimization
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