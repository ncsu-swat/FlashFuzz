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
        
        // Need at least basic parameters
        if (size < 32) return 0;
        
        // Extract dimensions and parameters
        uint32_t batch = (data[offset] % 4) + 1; offset++;
        uint32_t height = (data[offset] % 8) + 2; offset++;
        uint32_t width = (data[offset] % 8) + 2; offset++;
        uint32_t channels = (data[offset] % 4) + 1; offset++;
        
        uint32_t ksize_h = (data[offset] % 3) + 1; offset++;
        uint32_t ksize_w = (data[offset] % 3) + 1; offset++;
        uint32_t stride_h = (data[offset] % 2) + 1; offset++;
        uint32_t stride_w = (data[offset] % 2) + 1; offset++;
        
        bool padding_same = (data[offset] % 2) == 1; offset++;
        bool include_batch_in_index = (data[offset] % 2) == 1; offset++;
        
        // Calculate output dimensions for argmax
        uint32_t out_height, out_width;
        if (padding_same) {
            out_height = (height + stride_h - 1) / stride_h;
            out_width = (width + stride_w - 1) / stride_w;
        } else {
            out_height = (height - ksize_h) / stride_h + 1;
            out_width = (width - ksize_w) / stride_w + 1;
        }
        
        if (out_height == 0 || out_width == 0) return 0;
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input tensor
        tensorflow::TensorShape input_shape({batch, height, width, channels});
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, input_shape);
        auto input_flat = input_tensor.flat<float>();
        
        size_t input_size = batch * height * width * channels;
        for (size_t i = 0; i < input_size && offset < size; ++i, ++offset) {
            input_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
        }
        
        // Create grad tensor (same shape as input)
        tensorflow::Tensor grad_tensor(tensorflow::DT_FLOAT, input_shape);
        auto grad_flat = grad_tensor.flat<float>();
        
        for (size_t i = 0; i < input_size && offset < size; ++i, ++offset) {
            grad_flat(i) = static_cast<float>(data[offset % size]) / 255.0f - 0.5f;
        }
        
        // Create argmax tensor
        tensorflow::TensorShape argmax_shape({batch, out_height, out_width, channels});
        tensorflow::Tensor argmax_tensor(tensorflow::DT_INT64, argmax_shape);
        auto argmax_flat = argmax_tensor.flat<int64_t>();
        
        size_t argmax_size = batch * out_height * out_width * channels;
        int64_t max_index = include_batch_in_index ? 
            (batch * height * width * channels - 1) : 
            (height * width - 1);
            
        for (size_t i = 0; i < argmax_size && offset < size; ++i, ++offset) {
            argmax_flat(i) = static_cast<int64_t>(data[offset % size]) % (max_index + 1);
        }
        
        // Create input placeholders
        auto input_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto grad_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto argmax_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
        
        // Set up operation attributes
        std::vector<int> ksize = {1, static_cast<int>(ksize_h), static_cast<int>(ksize_w), 1};
        std::vector<int> strides = {1, static_cast<int>(stride_h), static_cast<int>(stride_w), 1};
        std::string padding = padding_same ? "SAME" : "VALID";
        
        // Create the MaxPoolGradGradWithArgmax operation
        auto max_pool_grad_grad = tensorflow::ops::MaxPoolGradGradWithArgmax(
            root,
            input_ph,
            grad_ph,
            argmax_ph,
            ksize,
            strides,
            padding,
            tensorflow::ops::MaxPoolGradGradWithArgmax::IncludeBatchInIndex(include_batch_in_index)
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run(
            {{input_ph, input_tensor}, {grad_ph, grad_tensor}, {argmax_ph, argmax_tensor}},
            {max_pool_grad_grad},
            &outputs
        );
        
        if (!status.ok()) {
            std::cout << "TensorFlow operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output
        if (!outputs.empty()) {
            const auto& output = outputs[0];
            if (output.shape() != input_shape) {
                std::cout << "Output shape mismatch" << std::endl;
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