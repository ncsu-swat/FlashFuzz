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
        
        // Extract dimensions from fuzz data
        int batch = (data[offset] % 4) + 1;
        offset++;
        int in_height = (data[offset] % 8) + 1;
        offset++;
        int in_width = (data[offset] % 8) + 1;
        offset++;
        int in_channels = (data[offset] % 4) + 1;
        offset++;
        int filter_height = (data[offset] % 4) + 1;
        offset++;
        int filter_width = (data[offset] % 4) + 1;
        offset++;
        int out_channels = (data[offset] % 4) + 1;
        offset++;
        
        // Extract stride values
        int stride_h = (data[offset] % 3) + 1;
        offset++;
        int stride_w = (data[offset] % 3) + 1;
        offset++;
        
        // Extract padding type
        std::string padding = (data[offset] % 2) ? "SAME" : "VALID";
        offset++;
        
        // Extract data format
        std::string data_format = (data[offset] % 2) ? "NHWC" : "NCHW";
        offset++;
        
        // Calculate output dimensions based on padding
        int out_height, out_width;
        if (padding == "SAME") {
            out_height = (in_height + stride_h - 1) / stride_h;
            out_width = (in_width + stride_w - 1) / stride_w;
        } else {
            out_height = (in_height - filter_height) / stride_h + 1;
            out_width = (in_width - filter_width) / stride_w + 1;
        }
        
        if (out_height <= 0 || out_width <= 0) return 0;
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input tensor
        tensorflow::TensorShape input_shape;
        if (data_format == "NHWC") {
            input_shape = tensorflow::TensorShape({batch, in_height, in_width, in_channels});
        } else {
            input_shape = tensorflow::TensorShape({batch, in_channels, in_height, in_width});
        }
        
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, input_shape);
        auto input_flat = input_tensor.flat<float>();
        
        // Fill input tensor with fuzz data
        size_t input_size = input_tensor.NumElements();
        for (int i = 0; i < input_size && offset < size; i++) {
            input_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
            offset++;
        }
        
        // Create filter_sizes tensor
        tensorflow::Tensor filter_sizes_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({4}));
        auto filter_sizes_flat = filter_sizes_tensor.flat<int32_t>();
        filter_sizes_flat(0) = filter_height;
        filter_sizes_flat(1) = filter_width;
        filter_sizes_flat(2) = in_channels;
        filter_sizes_flat(3) = out_channels;
        
        // Create out_backprop tensor
        tensorflow::TensorShape out_backprop_shape;
        if (data_format == "NHWC") {
            out_backprop_shape = tensorflow::TensorShape({batch, out_height, out_width, out_channels});
        } else {
            out_backprop_shape = tensorflow::TensorShape({batch, out_channels, out_height, out_width});
        }
        
        tensorflow::Tensor out_backprop_tensor(tensorflow::DT_FLOAT, out_backprop_shape);
        auto out_backprop_flat = out_backprop_tensor.flat<float>();
        
        // Fill out_backprop tensor with fuzz data
        size_t out_backprop_size = out_backprop_tensor.NumElements();
        for (int i = 0; i < out_backprop_size && offset < size; i++) {
            out_backprop_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
            offset++;
        }
        
        // Create input placeholders
        auto input_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto filter_sizes_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto out_backprop_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        // Set up Conv2DBackpropFilter operation attributes
        tensorflow::ops::Conv2DBackpropFilter::Attrs attrs;
        attrs = attrs.Padding(padding);
        attrs = attrs.DataFormat(data_format);
        attrs = attrs.UseCudnnOnGpu(true);
        attrs = attrs.Dilations({1, 1, 1, 1});
        
        // Create Conv2DBackpropFilter operation
        auto conv2d_backprop_filter = tensorflow::ops::Conv2DBackpropFilter(
            root, input_ph, filter_sizes_ph, out_backprop_ph, 
            {1, stride_h, stride_w, 1}, attrs);
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run(
            {{input_ph, input_tensor}, 
             {filter_sizes_ph, filter_sizes_tensor}, 
             {out_backprop_ph, out_backprop_tensor}},
            {conv2d_backprop_filter}, &outputs);
        
        if (!status.ok()) {
            std::cout << "Conv2DBackpropFilter failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output tensor properties
        if (!outputs.empty()) {
            const auto& output = outputs[0];
            if (output.dims() != 4) {
                std::cout << "Unexpected output dimensions: " << output.dims() << std::endl;
                return 0;
            }
            
            // Check output shape matches expected filter shape
            if (output.dim_size(0) != filter_height ||
                output.dim_size(1) != filter_width ||
                output.dim_size(2) != in_channels ||
                output.dim_size(3) != out_channels) {
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