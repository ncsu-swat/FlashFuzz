#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions from fuzz data
        int batch = (data[offset] % 4) + 1; offset++;
        int height = (data[offset] % 8) + 1; offset++;
        int width = (data[offset] % 8) + 1; offset++;
        int channels = (data[offset] % 8) + 1; offset++;
        
        // Extract parameters
        float epsilon = 0.0001f + (data[offset] % 100) * 0.00001f; offset++;
        bool is_training = data[offset] % 2; offset++;
        int data_format_idx = data[offset] % 2; offset++;
        std::string data_format = (data_format_idx == 0) ? "NHWC" : "NCHW";
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input tensors
        tensorflow::TensorShape input_shape;
        if (data_format == "NHWC") {
            input_shape = tensorflow::TensorShape({batch, height, width, channels});
        } else {
            input_shape = tensorflow::TensorShape({batch, channels, height, width});
        }
        
        tensorflow::TensorShape channel_shape({channels});
        
        // Create y_backprop tensor
        tensorflow::Tensor y_backprop_tensor(tensorflow::DT_FLOAT, input_shape);
        auto y_backprop_flat = y_backprop_tensor.flat<float>();
        for (int i = 0; i < y_backprop_flat.size() && offset < size; i++, offset++) {
            y_backprop_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
        }
        
        // Create x tensor
        tensorflow::Tensor x_tensor(tensorflow::DT_FLOAT, input_shape);
        auto x_flat = x_tensor.flat<float>();
        for (int i = 0; i < x_flat.size() && offset < size; i++, offset++) {
            x_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
        }
        
        // Create scale tensor
        tensorflow::Tensor scale_tensor(tensorflow::DT_FLOAT, channel_shape);
        auto scale_flat = scale_tensor.flat<float>();
        for (int i = 0; i < scale_flat.size() && offset < size; i++, offset++) {
            scale_flat(i) = 1.0f + static_cast<float>(data[offset % size]) / 255.0f;
        }
        
        // Create reserve_space tensors
        tensorflow::Tensor reserve_space_1_tensor(tensorflow::DT_FLOAT, channel_shape);
        auto reserve_space_1_flat = reserve_space_1_tensor.flat<float>();
        for (int i = 0; i < reserve_space_1_flat.size() && offset < size; i++, offset++) {
            reserve_space_1_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
        }
        
        tensorflow::Tensor reserve_space_2_tensor(tensorflow::DT_FLOAT, channel_shape);
        auto reserve_space_2_flat = reserve_space_2_tensor.flat<float>();
        for (int i = 0; i < reserve_space_2_flat.size() && offset < size; i++, offset++) {
            reserve_space_2_flat(i) = 1.0f + static_cast<float>(data[offset % size]) / 255.0f;
        }
        
        tensorflow::Tensor reserve_space_3_tensor(tensorflow::DT_FLOAT, channel_shape);
        auto reserve_space_3_flat = reserve_space_3_tensor.flat<float>();
        for (int i = 0; i < reserve_space_3_flat.size() && offset < size; i++, offset++) {
            reserve_space_3_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
        }
        
        // Create placeholder ops
        auto y_backprop = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto x = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto scale = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto reserve_space_1 = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto reserve_space_2 = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto reserve_space_3 = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        // Create FusedBatchNormGradV3 operation
        auto fused_batch_norm_grad = tensorflow::ops::FusedBatchNormGradV3(
            root,
            y_backprop,
            x,
            scale,
            reserve_space_1,
            reserve_space_2,
            reserve_space_3,
            tensorflow::ops::FusedBatchNormGradV3::Epsilon(epsilon)
                .DataFormat(data_format)
                .IsTraining(is_training)
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run(
            {{y_backprop, y_backprop_tensor},
             {x, x_tensor},
             {scale, scale_tensor},
             {reserve_space_1, reserve_space_1_tensor},
             {reserve_space_2, reserve_space_2_tensor},
             {reserve_space_3, reserve_space_3_tensor}},
            {fused_batch_norm_grad.x_backprop,
             fused_batch_norm_grad.scale_backprop,
             fused_batch_norm_grad.offset_backprop,
             fused_batch_norm_grad.reserve_space_4,
             fused_batch_norm_grad.reserve_space_5},
            &outputs
        );
        
        if (!status.ok()) {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
            return -1;
        }
        
        // Verify outputs
        if (outputs.size() != 5) {
            std::cout << "Expected 5 outputs, got " << outputs.size() << std::endl;
            return -1;
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}