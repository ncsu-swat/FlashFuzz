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
        if (size < 64) return 0;
        
        // Extract dimensions
        int batch = (data[offset] % 4) + 1; offset++;
        int in_height = (data[offset] % 8) + 1; offset++;
        int in_width = (data[offset] % 8) + 1; offset++;
        int in_channels = (data[offset] % 4) + 1; offset++;
        int filter_height = (data[offset] % 4) + 1; offset++;
        int filter_width = (data[offset] % 4) + 1; offset++;
        int out_channels = (data[offset] % 4) + 1; offset++;
        
        // Extract strides
        int stride_h = (data[offset] % 3) + 1; offset++;
        int stride_w = (data[offset] % 3) + 1; offset++;
        
        // Extract padding type
        std::string padding;
        int padding_type = data[offset] % 3; offset++;
        switch (padding_type) {
            case 0: padding = "VALID"; break;
            case 1: padding = "SAME"; break;
            case 2: padding = "EXPLICIT"; break;
        }
        
        // Extract data format
        std::string data_format = (data[offset] % 2) ? "NCHW" : "NHWC"; offset++;
        
        // Calculate output dimensions for NHWC format
        int out_height, out_width;
        if (padding == "VALID") {
            out_height = (in_height - filter_height) / stride_h + 1;
            out_width = (in_width - filter_width) / stride_w + 1;
        } else {
            out_height = (in_height + stride_h - 1) / stride_h;
            out_width = (in_width + stride_w - 1) / stride_w;
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
        for (int i = 0; i < input_flat.size() && offset < size; i++, offset++) {
            input_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
        }
        
        // Create filter tensor (only shape matters for backprop)
        tensorflow::TensorShape filter_shape({filter_height, filter_width, in_channels, out_channels});
        tensorflow::Tensor filter_tensor(tensorflow::DT_FLOAT, filter_shape);
        auto filter_flat = filter_tensor.flat<float>();
        
        // Fill filter tensor
        for (int i = 0; i < filter_flat.size() && offset < size; i++, offset++) {
            filter_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
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
        
        // Fill out_backprop tensor
        for (int i = 0; i < out_backprop_flat.size() && offset < size; i++, offset++) {
            out_backprop_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
        }
        
        // Create placeholder ops
        auto input_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto filter_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto out_backprop_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        // Set up operation attributes
        tensorflow::ops::Conv2DBackpropFilter::Attrs attrs;
        attrs = attrs.UseCudnnOnGpu(false);
        attrs = attrs.DataFormat(data_format);
        attrs = attrs.Dilations({1, 1, 1, 1});
        
        std::vector<int> strides_vec;
        if (data_format == "NHWC") {
            strides_vec = {1, stride_h, stride_w, 1};
        } else {
            strides_vec = {1, 1, stride_h, stride_w};
        }
        
        // Handle explicit padding
        if (padding == "EXPLICIT") {
            std::vector<int> explicit_paddings = {0, 0, 1, 1, 1, 1, 0, 0};
            attrs = attrs.ExplicitPaddings(explicit_paddings);
        }
        
        // Create the operation
        auto conv2d_backprop_filter = tensorflow::ops::Conv2DBackpropFilter(
            root, input_ph, filter_ph, out_backprop_ph, strides_vec, padding, attrs);
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run(
            {{input_ph, input_tensor}, {filter_ph, filter_tensor}, {out_backprop_ph, out_backprop_tensor}},
            {conv2d_backprop_filter}, &outputs);
        
        if (!status.ok()) {
            std::cout << "TensorFlow operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output shape
        if (!outputs.empty()) {
            const auto& output_shape = outputs[0].shape();
            if (output_shape.dims() != 4) {
                std::cout << "Unexpected output dimensions: " << output_shape.dims() << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}