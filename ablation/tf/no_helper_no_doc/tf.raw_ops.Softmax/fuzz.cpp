#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/common_runtime/kernel_benchmark_testlib.h>
#include <tensorflow/core/framework/fake_input.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/types.pb.h>
#include <tensorflow/core/kernels/ops_testutil.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions
        int32_t batch_size = *reinterpret_cast<const int32_t*>(data + offset) % 10 + 1;
        offset += 4;
        int32_t num_classes = *reinterpret_cast<const int32_t*>(data + offset) % 100 + 1;
        offset += 4;
        
        // Calculate required data size
        size_t required_size = batch_size * num_classes * sizeof(float);
        if (offset + required_size > size) return 0;
        
        // Create input tensor
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, 
                                      tensorflow::TensorShape({batch_size, num_classes}));
        
        auto input_flat = input_tensor.flat<float>();
        
        // Fill tensor with fuzz data
        const float* float_data = reinterpret_cast<const float*>(data + offset);
        for (int i = 0; i < batch_size * num_classes; ++i) {
            if (offset + (i + 1) * sizeof(float) <= size) {
                float val = float_data[i];
                // Clamp values to prevent overflow/underflow
                if (std::isnan(val) || std::isinf(val)) {
                    val = 0.0f;
                } else if (val > 100.0f) {
                    val = 100.0f;
                } else if (val < -100.0f) {
                    val = -100.0f;
                }
                input_flat(i) = val;
            } else {
                input_flat(i) = 0.0f;
            }
        }
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create graph def
        tensorflow::GraphDef graph_def;
        
        // Add input placeholder
        tensorflow::NodeDef* input_node = graph_def.add_node();
        input_node->set_name("input");
        input_node->set_op("Placeholder");
        (*input_node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        (*input_node->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(batch_size);
        (*input_node->mutable_attr())["shape"].mutable_shape()->add_dim()->set_size(num_classes);
        
        // Add Softmax node
        tensorflow::NodeDef* softmax_node = graph_def.add_node();
        softmax_node->set_name("softmax");
        softmax_node->set_op("Softmax");
        softmax_node->add_input("input");
        (*softmax_node->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
        
        // Create session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the softmax operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({{"input", input_tensor}}, {"softmax"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Verify output shape matches input shape
            if (outputs[0].shape().dims() == 2 && 
                outputs[0].shape().dim_size(0) == batch_size &&
                outputs[0].shape().dim_size(1) == num_classes) {
                
                auto output_flat = outputs[0].flat<float>();
                
                // Basic validation: check if probabilities sum to ~1 for each batch
                for (int b = 0; b < batch_size; ++b) {
                    float sum = 0.0f;
                    bool valid = true;
                    
                    for (int c = 0; c < num_classes; ++c) {
                        float val = output_flat(b * num_classes + c);
                        if (std::isnan(val) || std::isinf(val) || val < 0.0f || val > 1.0f) {
                            valid = false;
                            break;
                        }
                        sum += val;
                    }
                    
                    // Check if sum is approximately 1.0 (allowing for floating point errors)
                    if (valid && (sum < 0.99f || sum > 1.01f)) {
                        valid = false;
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