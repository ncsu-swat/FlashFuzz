#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/public/session.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < sizeof(bool)) {
            return 0;
        }
        
        // Extract boolean value from fuzzer input
        bool input_value = (data[offset] % 2) == 1;
        offset += sizeof(bool);
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input tensor (scalar boolean)
        tensorflow::Tensor input_tensor(tensorflow::DT_BOOL, tensorflow::TensorShape({}));
        input_tensor.scalar<bool>()() = input_value;
        
        // Create placeholder for input
        auto input_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_BOOL);
        
        // Create LoopCond operation
        auto loop_cond = tensorflow::ops::LoopCond(root, input_placeholder);
        
        // Create session
        tensorflow::ClientSession session(root);
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({{input_placeholder, input_tensor}}, {loop_cond}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Verify output is boolean scalar
            if (outputs[0].dtype() == tensorflow::DT_BOOL && 
                outputs[0].shape().dims() == 0) {
                bool output_value = outputs[0].scalar<bool>()();
                // LoopCond should forward input to output
                if (output_value == input_value) {
                    // Test passed
                }
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}