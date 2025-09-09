#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions for input tensor
        uint32_t batch = *reinterpret_cast<const uint32_t*>(data + offset) % 8 + 1;
        offset += 4;
        uint32_t height = *reinterpret_cast<const uint32_t*>(data + offset) % 64 + 4;
        offset += 4;
        uint32_t width = *reinterpret_cast<const uint32_t*>(data + offset) % 64 + 4;
        offset += 4;
        uint32_t channels = *reinterpret_cast<const uint32_t*>(data + offset) % 16 + 1;
        offset += 4;
        
        // Extract pooling ratios
        float pooling_ratio_h = 1.0f + ((*reinterpret_cast<const uint32_t*>(data + offset) % 100) / 100.0f);
        offset += 4;
        float pooling_ratio_w = 1.0f + ((*reinterpret_cast<const uint32_t*>(data + offset) % 100) / 100.0f);
        offset += 4;
        
        // Extract boolean flags
        bool pseudo_random = (data[offset] % 2) == 1;
        offset += 1;
        bool overlapping = (data[offset] % 2) == 1;
        offset += 1;
        bool deterministic = (data[offset] % 2) == 1;
        offset += 1;
        
        // Extract seeds
        int32_t seed = *reinterpret_cast<const int32_t*>(data + offset);
        offset += 4;
        int32_t seed2 = *reinterpret_cast<const int32_t*>(data + offset);
        offset += 4;
        
        if (offset >= size) return 0;
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input tensor shape
        tensorflow::TensorShape input_shape({static_cast<int64_t>(batch), 
                                           static_cast<int64_t>(height), 
                                           static_cast<int64_t>(width), 
                                           static_cast<int64_t>(channels)});
        
        // Create input tensor with float32 type
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, input_shape);
        auto input_flat = input_tensor.flat<float>();
        
        // Fill tensor with data from fuzzer input
        size_t tensor_size = batch * height * width * channels;
        for (size_t i = 0; i < tensor_size && offset < size; ++i) {
            if (offset + 4 <= size) {
                input_flat(i) = *reinterpret_cast<const float*>(data + offset);
                offset += 4;
            } else {
                input_flat(i) = static_cast<float>(data[offset % size]);
                offset++;
            }
        }
        
        // Create input placeholder
        auto input_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        // Set up pooling ratio vector
        std::vector<float> pooling_ratios = {1.0f, pooling_ratio_h, pooling_ratio_w, 1.0f};
        
        // Create FractionalAvgPool operation
        auto fractional_avg_pool = tensorflow::ops::FractionalAvgPool(
            root, 
            input_placeholder,
            pooling_ratios,
            tensorflow::ops::FractionalAvgPool::PseudoRandom(pseudo_random)
                .Overlapping(overlapping)
                .Deterministic(deterministic)
                .Seed(seed)
                .Seed2(seed2)
        );
        
        // Create session
        tensorflow::SessionOptions session_options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(session_options));
        
        // Create graph def
        tensorflow::GraphDef graph_def;
        tensorflow::Status status = root.ToGraphDef(&graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Create session and run
        status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({{input_placeholder.node()->name(), input_tensor}},
                             {fractional_avg_pool.output.node()->name(),
                              fractional_avg_pool.row_pooling_sequence.node()->name(),
                              fractional_avg_pool.col_pooling_sequence.node()->name()},
                             {}, &outputs);
        
        if (status.ok() && outputs.size() == 3) {
            // Verify output shapes are reasonable
            auto output_shape = outputs[0].shape();
            auto row_seq_shape = outputs[1].shape();
            auto col_seq_shape = outputs[2].shape();
            
            if (output_shape.dims() == 4 && 
                row_seq_shape.dims() >= 1 && 
                col_seq_shape.dims() >= 1) {
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