#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/kernels/invert_permutation_op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/strcat.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/common_runtime/direct_session.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < sizeof(int32_t)) {
            return 0;
        }
        
        // Extract dimension size
        int32_t dim_size;
        memcpy(&dim_size, data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        
        // Clamp dimension size to reasonable bounds
        dim_size = std::abs(dim_size) % 1000 + 1;
        
        if (offset + dim_size * sizeof(int32_t) > size) {
            return 0;
        }
        
        // Create input tensor
        tensorflow::Tensor input_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({dim_size}));
        auto input_flat = input_tensor.flat<int32_t>();
        
        // Fill tensor with data from fuzzer input
        for (int i = 0; i < dim_size; i++) {
            int32_t value;
            if (offset + sizeof(int32_t) <= size) {
                memcpy(&value, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
            } else {
                value = i; // fallback to valid permutation
            }
            // Ensure values are within valid range for permutation
            input_flat(i) = std::abs(value) % dim_size;
        }
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Build graph
        tensorflow::GraphDef graph_def;
        tensorflow::GraphDefBuilder builder(tensorflow::GraphDefBuilder::kFailImmediately);
        
        auto input_node = tensorflow::ops::Const(input_tensor, builder.opts().WithName("input"));
        auto invert_perm = tensorflow::ops::InvertPermutation(input_node, builder.opts().WithName("invert_perm"));
        
        tensorflow::Status status = builder.ToGraphDef(&graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({}, {"invert_perm:0"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Verify output tensor properties
            const tensorflow::Tensor& output = outputs[0];
            if (output.dtype() == tensorflow::DT_INT32 && 
                output.shape().dims() == 1 && 
                output.shape().dim_size(0) == dim_size) {
                auto output_flat = output.flat<int32_t>();
                // Basic validation that output values are in valid range
                for (int i = 0; i < dim_size; i++) {
                    if (output_flat(i) < 0 || output_flat(i) >= dim_size) {
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