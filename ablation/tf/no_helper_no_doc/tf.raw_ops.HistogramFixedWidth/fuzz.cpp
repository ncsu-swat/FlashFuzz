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
        int32_t values_size = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        values_size = std::abs(values_size) % 1000 + 1; // Limit size and ensure positive
        
        int32_t nbins = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        nbins = std::abs(nbins) % 100 + 1; // Limit bins and ensure positive
        
        float range_min = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        
        float range_max = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        
        // Ensure valid range
        if (range_min >= range_max) {
            range_max = range_min + 1.0f;
        }
        
        // Calculate remaining data for values
        size_t remaining_size = size - offset;
        size_t available_floats = remaining_size / sizeof(float);
        
        if (available_floats == 0) return 0;
        
        // Adjust values_size based on available data
        values_size = std::min(values_size, static_cast<int32_t>(available_floats));
        
        // Create TensorFlow session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create graph
        tensorflow::GraphDef graph_def;
        
        // Create placeholder for values
        tensorflow::NodeDef values_placeholder;
        values_placeholder.set_name("values");
        values_placeholder.set_op("Placeholder");
        (*values_placeholder.mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        (*values_placeholder.mutable_attr())["shape"].mutable_shape();
        *graph_def.add_node() = values_placeholder;
        
        // Create placeholder for value_range
        tensorflow::NodeDef range_placeholder;
        range_placeholder.set_name("value_range");
        range_placeholder.set_op("Placeholder");
        (*range_placeholder.mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        (*range_placeholder.mutable_attr())["shape"].mutable_shape();
        *graph_def.add_node() = range_placeholder;
        
        // Create HistogramFixedWidth node
        tensorflow::NodeDef histogram_node;
        histogram_node.set_name("histogram");
        histogram_node.set_op("HistogramFixedWidth");
        histogram_node.add_input("values");
        histogram_node.add_input("value_range");
        (*histogram_node.mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
        (*histogram_node.mutable_attr())["nbins"].set_i(nbins);
        *graph_def.add_node() = histogram_node;
        
        // Create session and add graph
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Prepare input tensors
        tensorflow::Tensor values_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({values_size}));
        auto values_flat = values_tensor.flat<float>();
        
        // Fill values tensor with fuzzer data
        for (int i = 0; i < values_size && offset + sizeof(float) <= size; ++i) {
            values_flat(i) = *reinterpret_cast<const float*>(data + offset);
            offset += sizeof(float);
        }
        
        // Create range tensor
        tensorflow::Tensor range_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({2}));
        auto range_flat = range_tensor.flat<float>();
        range_flat(0) = range_min;
        range_flat(1) = range_max;
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({{"values", values_tensor}, {"value_range", range_tensor}},
                             {"histogram"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Verify output shape and basic properties
            const tensorflow::Tensor& output = outputs[0];
            if (output.dtype() == tensorflow::DT_INT32 && 
                output.shape().dims() == 1 && 
                output.shape().dim_size(0) == nbins) {
                // Basic validation passed
                auto output_flat = output.flat<int32_t>();
                int32_t total_count = 0;
                for (int i = 0; i < nbins; ++i) {
                    if (output_flat(i) >= 0) {
                        total_count += output_flat(i);
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