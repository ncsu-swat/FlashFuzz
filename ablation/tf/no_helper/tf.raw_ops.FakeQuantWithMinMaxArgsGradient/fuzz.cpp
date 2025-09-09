#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/kernels/fake_quant_ops_functor.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/cc/framework/ops.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract parameters from fuzzer input
        int32_t batch_size = *reinterpret_cast<const int32_t*>(data + offset) % 10 + 1;
        offset += sizeof(int32_t);
        
        int32_t height = *reinterpret_cast<const int32_t*>(data + offset) % 10 + 1;
        offset += sizeof(int32_t);
        
        int32_t width = *reinterpret_cast<const int32_t*>(data + offset) % 10 + 1;
        offset += sizeof(int32_t);
        
        int32_t channels = *reinterpret_cast<const int32_t*>(data + offset) % 10 + 1;
        offset += sizeof(int32_t);
        
        float min_val = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        
        float max_val = *reinterpret_cast<const float*>(data + offset);
        offset += sizeof(float);
        
        int32_t num_bits = (*reinterpret_cast<const int32_t*>(data + offset) % 16) + 1;
        offset += sizeof(int32_t);
        
        bool narrow_range = (*reinterpret_cast<const uint8_t*>(data + offset)) % 2;
        offset += sizeof(uint8_t);
        
        // Ensure min < max
        if (min_val >= max_val) {
            max_val = min_val + 1.0f;
        }
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input tensors
        tensorflow::TensorShape shape({batch_size, height, width, channels});
        tensorflow::Tensor gradients_tensor(tensorflow::DT_FLOAT, shape);
        tensorflow::Tensor inputs_tensor(tensorflow::DT_FLOAT, shape);
        
        // Fill tensors with data from fuzzer input
        auto gradients_flat = gradients_tensor.flat<float>();
        auto inputs_flat = inputs_tensor.flat<float>();
        
        size_t tensor_size = batch_size * height * width * channels;
        for (size_t i = 0; i < tensor_size && offset + sizeof(float) <= size; ++i) {
            gradients_flat(i) = *reinterpret_cast<const float*>(data + offset);
            offset += sizeof(float);
            if (offset + sizeof(float) <= size) {
                inputs_flat(i) = *reinterpret_cast<const float*>(data + offset);
                offset += sizeof(float);
            } else {
                inputs_flat(i) = 0.0f;
            }
        }
        
        // Fill remaining elements with zeros if not enough data
        for (size_t i = (offset - 8 * sizeof(int32_t) - sizeof(float) - sizeof(uint8_t)) / (2 * sizeof(float)); i < tensor_size; ++i) {
            if (i < gradients_flat.size()) gradients_flat(i) = 0.0f;
            if (i < inputs_flat.size()) inputs_flat(i) = 0.0f;
        }
        
        // Create placeholder ops
        auto gradients_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto inputs_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        // Create the FakeQuantWithMinMaxArgsGradient operation
        auto fake_quant_grad = tensorflow::ops::FakeQuantWithMinMaxArgsGradient(
            root, gradients_ph, inputs_ph,
            tensorflow::ops::FakeQuantWithMinMaxArgsGradient::Min(min_val)
                .Max(max_val)
                .NumBits(num_bits)
                .NarrowRange(narrow_range)
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run(
            {{gradients_ph, gradients_tensor}, {inputs_ph, inputs_tensor}},
            {fake_quant_grad},
            &outputs
        );
        
        if (!status.ok()) {
            std::cout << "TensorFlow operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output
        if (!outputs.empty() && outputs[0].dtype() == tensorflow::DT_FLOAT) {
            auto output_flat = outputs[0].flat<float>();
            // Basic sanity check - ensure no NaN values
            for (int i = 0; i < output_flat.size(); ++i) {
                if (std::isnan(output_flat(i))) {
                    std::cout << "NaN detected in output" << std::endl;
                    break;
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