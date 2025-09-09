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
#include <tensorflow/cc/ops/standard_ops.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 20) return 0; // Need minimum data for parameters
        
        // Extract parameters from fuzz input
        int num_bits = (data[offset] % 7) + 2; // 2-8 range
        offset++;
        
        bool narrow_range = data[offset] % 2;
        offset++;
        
        // Extract tensor dimensions
        if (offset + 8 >= size) return 0;
        int batch_size = std::max(1, static_cast<int>(data[offset] % 10) + 1);
        offset++;
        int height = std::max(1, static_cast<int>(data[offset] % 10) + 1);
        offset++;
        int width = std::max(1, static_cast<int>(data[offset] % 10) + 1);
        offset++;
        int channels = std::max(1, static_cast<int>(data[offset] % 10) + 1);
        offset++;
        
        // Calculate required data size
        int tensor_size = batch_size * height * width * channels;
        int required_size = tensor_size * 2 * sizeof(float) + 2 * sizeof(float); // gradients + inputs + min + max
        
        if (offset + required_size > size) return 0;
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input tensors
        tensorflow::TensorShape shape({batch_size, height, width, channels});
        
        // Create gradients tensor
        tensorflow::Tensor gradients_tensor(tensorflow::DT_FLOAT, shape);
        auto gradients_flat = gradients_tensor.flat<float>();
        std::memcpy(gradients_flat.data(), data + offset, tensor_size * sizeof(float));
        offset += tensor_size * sizeof(float);
        
        // Create inputs tensor
        tensorflow::Tensor inputs_tensor(tensorflow::DT_FLOAT, shape);
        auto inputs_flat = inputs_tensor.flat<float>();
        std::memcpy(inputs_flat.data(), data + offset, tensor_size * sizeof(float));
        offset += tensor_size * sizeof(float);
        
        // Create min tensor (scalar)
        tensorflow::Tensor min_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        float min_val;
        std::memcpy(&min_val, data + offset, sizeof(float));
        min_tensor.scalar<float>()() = min_val;
        offset += sizeof(float);
        
        // Create max tensor (scalar)
        tensorflow::Tensor max_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        float max_val;
        std::memcpy(&max_val, data + offset, sizeof(float));
        max_tensor.scalar<float>()() = max_val;
        
        // Ensure min < max
        if (min_val >= max_val) {
            max_val = min_val + 1.0f;
            max_tensor.scalar<float>()() = max_val;
        }
        
        // Create placeholder ops
        auto gradients_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto inputs_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto min_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto max_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        // Create the FakeQuantWithMinMaxVarsGradient operation
        auto fake_quant_grad = tensorflow::ops::FakeQuantWithMinMaxVarsGradient(
            root, gradients_ph, inputs_ph, min_ph, max_ph,
            tensorflow::ops::FakeQuantWithMinMaxVarsGradient::NumBits(num_bits)
                .NarrowRange(narrow_range));
        
        // Create session
        tensorflow::ClientSession session(root);
        
        // Prepare feed dict
        std::vector<std::pair<tensorflow::Output, tensorflow::Tensor>> feed_dict = {
            {gradients_ph, gradients_tensor},
            {inputs_ph, inputs_tensor},
            {min_ph, min_tensor},
            {max_ph, max_tensor}
        };
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        auto status = session.Run(feed_dict, 
            {fake_quant_grad.backprops_wrt_input, 
             fake_quant_grad.backprop_wrt_min, 
             fake_quant_grad.backprop_wrt_max}, 
            &outputs);
        
        if (!status.ok()) {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output shapes
        if (outputs.size() != 3) {
            std::cout << "Expected 3 outputs, got " << outputs.size() << std::endl;
            return 0;
        }
        
        // Check that backprops_wrt_input has same shape as input
        if (outputs[0].shape() != shape) {
            std::cout << "backprops_wrt_input shape mismatch" << std::endl;
            return 0;
        }
        
        // Check that backprop_wrt_min and backprop_wrt_max are scalars
        if (outputs[1].dims() != 0 || outputs[2].dims() != 0) {
            std::cout << "min/max gradients should be scalars" << std::endl;
            return 0;
        }
        
        // Verify all outputs are finite
        auto check_finite = [](const tensorflow::Tensor& t) {
            auto flat = t.flat<float>();
            for (int i = 0; i < flat.size(); ++i) {
                if (!std::isfinite(flat(i))) {
                    return false;
                }
            }
            return true;
        };
        
        for (const auto& output : outputs) {
            if (!check_finite(output)) {
                std::cout << "Non-finite values in output" << std::endl;
                return 0;
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}