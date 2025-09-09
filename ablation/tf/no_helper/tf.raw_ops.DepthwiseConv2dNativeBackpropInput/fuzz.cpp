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
        
        if (size < 64) return 0;
        
        // Extract basic parameters
        int batch = (data[offset] % 4) + 1;
        offset++;
        int height = (data[offset] % 8) + 3;
        offset++;
        int width = (data[offset] % 8) + 3;
        offset++;
        int in_channels = (data[offset] % 4) + 1;
        offset++;
        int filter_height = (data[offset] % 3) + 1;
        offset++;
        int filter_width = (data[offset] % 3) + 1;
        offset++;
        int depth_multiplier = (data[offset] % 3) + 1;
        offset++;
        
        // Extract stride parameters
        int stride_h = (data[offset] % 3) + 1;
        offset++;
        int stride_w = (data[offset] % 3) + 1;
        offset++;
        
        // Extract padding type
        std::string padding = (data[offset] % 2 == 0) ? "SAME" : "VALID";
        offset++;
        
        // Extract data format
        std::string data_format = (data[offset] % 2 == 0) ? "NHWC" : "NCHW";
        offset++;
        
        // Calculate output dimensions
        int out_channels = in_channels * depth_multiplier;
        int out_height, out_width;
        
        if (padding == "SAME") {
            out_height = (height + stride_h - 1) / stride_h;
            out_width = (width + stride_w - 1) / stride_w;
        } else {
            out_height = (height - filter_height) / stride_h + 1;
            out_width = (width - filter_width) / stride_w + 1;
        }
        
        if (out_height <= 0 || out_width <= 0) return 0;
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input_sizes tensor
        tensorflow::Tensor input_sizes_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({4}));
        auto input_sizes_flat = input_sizes_tensor.flat<int32_t>();
        if (data_format == "NHWC") {
            input_sizes_flat(0) = batch;
            input_sizes_flat(1) = height;
            input_sizes_flat(2) = width;
            input_sizes_flat(3) = in_channels;
        } else {
            input_sizes_flat(0) = batch;
            input_sizes_flat(1) = in_channels;
            input_sizes_flat(2) = height;
            input_sizes_flat(3) = width;
        }
        
        // Create filter tensor
        tensorflow::Tensor filter_tensor(tensorflow::DT_FLOAT, 
            tensorflow::TensorShape({filter_height, filter_width, in_channels, depth_multiplier}));
        auto filter_flat = filter_tensor.flat<float>();
        for (int i = 0; i < filter_flat.size() && offset < size; i++, offset++) {
            filter_flat(i) = static_cast<float>(data[offset]) / 255.0f - 0.5f;
        }
        
        // Create out_backprop tensor
        tensorflow::TensorShape out_backprop_shape;
        if (data_format == "NHWC") {
            out_backprop_shape = tensorflow::TensorShape({batch, out_height, out_width, out_channels});
        } else {
            out_backprop_shape = tensorflow::TensorShape({batch, out_channels, out_height, out_width});
        }
        
        tensorflow::Tensor out_backprop_tensor(tensorflow::DT_FLOAT, out_backprop_shape);
        auto out_backprop_flat = out_backprop_tensor.flat<float>();
        for (int i = 0; i < out_backprop_flat.size() && offset < size; i++, offset++) {
            out_backprop_flat(i) = static_cast<float>(data[offset]) / 255.0f - 0.5f;
        }
        
        // Create constant ops
        auto input_sizes_op = tensorflow::ops::Const(root, input_sizes_tensor);
        auto filter_op = tensorflow::ops::Const(root, filter_tensor);
        auto out_backprop_op = tensorflow::ops::Const(root, out_backprop_tensor);
        
        // Set up operation attributes
        tensorflow::ops::DepthwiseConv2dNativeBackpropInput::Attrs attrs;
        attrs = attrs.DataFormat(data_format);
        attrs = attrs.Dilations({1, 1, 1, 1});
        
        // Create the operation
        auto depthwise_backprop = tensorflow::ops::DepthwiseConv2dNativeBackpropInput(
            root, input_sizes_op, filter_op, out_backprop_op,
            {1, stride_h, stride_w, 1}, padding, attrs);
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({depthwise_backprop}, &outputs);
        
        if (!status.ok()) {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output shape
        if (!outputs.empty()) {
            const auto& output_shape = outputs[0].shape();
            if (data_format == "NHWC") {
                if (output_shape.dims() == 4 &&
                    output_shape.dim_size(0) == batch &&
                    output_shape.dim_size(1) == height &&
                    output_shape.dim_size(2) == width &&
                    output_shape.dim_size(3) == in_channels) {
                    // Shape is correct
                }
            } else {
                if (output_shape.dims() == 4 &&
                    output_shape.dim_size(0) == batch &&
                    output_shape.dim_size(1) == in_channels &&
                    output_shape.dim_size(2) == height &&
                    output_shape.dim_size(3) == width) {
                    // Shape is correct
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