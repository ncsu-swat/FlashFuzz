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
        
        // Extract dimensions for input_sizes (4 values for NHWC)
        int32_t input_batch = *reinterpret_cast<const int32_t*>(data + offset) % 4 + 1;
        offset += 4;
        int32_t input_height = *reinterpret_cast<const int32_t*>(data + offset) % 32 + 1;
        offset += 4;
        int32_t input_width = *reinterpret_cast<const int32_t*>(data + offset) % 32 + 1;
        offset += 4;
        int32_t input_channels = *reinterpret_cast<const int32_t*>(data + offset) % 16 + 1;
        offset += 4;
        
        // Extract filter dimensions
        int32_t filter_height = *reinterpret_cast<const int32_t*>(data + offset) % 8 + 1;
        offset += 4;
        int32_t filter_width = *reinterpret_cast<const int32_t*>(data + offset) % 8 + 1;
        offset += 4;
        int32_t depth_multiplier = *reinterpret_cast<const int32_t*>(data + offset) % 4 + 1;
        offset += 4;
        
        // Extract strides
        int32_t stride_h = *reinterpret_cast<const int32_t*>(data + offset) % 4 + 1;
        offset += 4;
        
        if (offset >= size) return 0;
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input_sizes tensor
        tensorflow::Tensor input_sizes_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({4}));
        auto input_sizes_flat = input_sizes_tensor.flat<int32_t>();
        input_sizes_flat(0) = input_batch;
        input_sizes_flat(1) = input_height;
        input_sizes_flat(2) = input_width;
        input_sizes_flat(3) = input_channels;
        
        auto input_sizes = tensorflow::ops::Const(root, input_sizes_tensor);
        
        // Calculate output dimensions for filter tensor
        int32_t out_height = (input_height + stride_h - 1) / stride_h;
        int32_t out_width = (input_width + stride_h - 1) / stride_h;
        int32_t out_channels = input_channels * depth_multiplier;
        
        // Create filter tensor
        tensorflow::Tensor filter_tensor(tensorflow::DT_FLOAT, 
            tensorflow::TensorShape({filter_height, filter_width, input_channels, depth_multiplier}));
        auto filter_flat = filter_tensor.flat<float>();
        
        size_t filter_size = filter_height * filter_width * input_channels * depth_multiplier;
        for (size_t i = 0; i < filter_size && offset + 4 <= size; ++i) {
            filter_flat(i) = *reinterpret_cast<const float*>(data + offset);
            offset += 4;
        }
        
        auto filter = tensorflow::ops::Const(root, filter_tensor);
        
        // Create out_backprop tensor
        tensorflow::Tensor out_backprop_tensor(tensorflow::DT_FLOAT,
            tensorflow::TensorShape({input_batch, out_height, out_width, out_channels}));
        auto out_backprop_flat = out_backprop_tensor.flat<float>();
        
        size_t out_backprop_size = input_batch * out_height * out_width * out_channels;
        for (size_t i = 0; i < out_backprop_size && offset + 4 <= size; ++i) {
            out_backprop_flat(i) = *reinterpret_cast<const float*>(data + offset);
            offset += 4;
        }
        
        auto out_backprop = tensorflow::ops::Const(root, out_backprop_tensor);
        
        // Set up operation attributes
        std::vector<int> strides = {1, stride_h, stride_h, 1};
        std::string padding = "SAME";
        std::string data_format = "NHWC";
        std::vector<int> dilations = {1, 1, 1, 1};
        
        // Create the DepthwiseConv2dNativeBackpropInput operation
        auto depthwise_conv2d_backprop_input = tensorflow::ops::DepthwiseConv2dNativeBackpropInput(
            root, input_sizes, filter, out_backprop,
            tensorflow::ops::DepthwiseConv2dNativeBackpropInput::Strides(strides)
                .Padding(padding)
                .DataFormat(data_format)
                .Dilations(dilations)
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({depthwise_conv2d_backprop_input}, &outputs);
        
        if (!status.ok()) {
            std::cout << "TensorFlow operation failed: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}