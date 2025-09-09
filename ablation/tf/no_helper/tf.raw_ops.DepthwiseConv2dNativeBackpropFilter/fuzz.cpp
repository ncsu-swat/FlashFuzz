#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/framework/graph.pb.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 64) return 0;
        
        // Extract basic parameters
        uint32_t batch = (data[offset] % 4) + 1;
        offset++;
        uint32_t in_height = (data[offset] % 8) + 1;
        offset++;
        uint32_t in_width = (data[offset] % 8) + 1;
        offset++;
        uint32_t in_channels = (data[offset] % 4) + 1;
        offset++;
        uint32_t filter_height = (data[offset] % 4) + 1;
        offset++;
        uint32_t filter_width = (data[offset] % 4) + 1;
        offset++;
        uint32_t depth_multiplier = (data[offset] % 4) + 1;
        offset++;
        
        // Extract strides
        uint32_t stride_h = (data[offset] % 3) + 1;
        offset++;
        uint32_t stride_w = (data[offset] % 3) + 1;
        offset++;
        
        // Extract data format (0 = NHWC, 1 = NCHW)
        bool is_nchw = (data[offset] % 2) == 1;
        offset++;
        
        // Extract padding type (0 = VALID, 1 = SAME)
        bool use_same_padding = (data[offset] % 2) == 1;
        offset++;
        
        // Calculate output dimensions
        uint32_t out_channels = in_channels * depth_multiplier;
        uint32_t out_height, out_width;
        
        if (use_same_padding) {
            out_height = (in_height + stride_h - 1) / stride_h;
            out_width = (in_width + stride_w - 1) / stride_w;
        } else {
            out_height = (in_height - filter_height) / stride_h + 1;
            out_width = (in_width - filter_width) / stride_w + 1;
        }
        
        if (out_height == 0 || out_width == 0) return 0;
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input tensor shape
        tensorflow::TensorShape input_shape;
        if (is_nchw) {
            input_shape = tensorflow::TensorShape({batch, in_channels, in_height, in_width});
        } else {
            input_shape = tensorflow::TensorShape({batch, in_height, in_width, in_channels});
        }
        
        // Create filter_sizes tensor
        tensorflow::Tensor filter_sizes_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({4}));
        auto filter_sizes_flat = filter_sizes_tensor.flat<int32_t>();
        filter_sizes_flat(0) = filter_height;
        filter_sizes_flat(1) = filter_width;
        filter_sizes_flat(2) = in_channels;
        filter_sizes_flat(3) = depth_multiplier;
        
        // Create out_backprop tensor shape
        tensorflow::TensorShape out_backprop_shape;
        if (is_nchw) {
            out_backprop_shape = tensorflow::TensorShape({batch, out_channels, out_height, out_width});
        } else {
            out_backprop_shape = tensorflow::TensorShape({batch, out_height, out_width, out_channels});
        }
        
        // Create input tensor with random data
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, input_shape);
        auto input_flat = input_tensor.flat<float>();
        for (int i = 0; i < input_flat.size() && offset < size; i++, offset++) {
            input_flat(i) = static_cast<float>(data[offset % size]) / 255.0f - 0.5f;
        }
        
        // Create out_backprop tensor with random data
        tensorflow::Tensor out_backprop_tensor(tensorflow::DT_FLOAT, out_backprop_shape);
        auto out_backprop_flat = out_backprop_tensor.flat<float>();
        for (int i = 0; i < out_backprop_flat.size() && offset < size; i++, offset++) {
            out_backprop_flat(i) = static_cast<float>(data[offset % size]) / 255.0f - 0.5f;
        }
        
        // Create placeholder ops
        auto input_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto filter_sizes_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto out_backprop_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        // Set up operation attributes
        tensorflow::ops::DepthwiseConv2dNativeBackpropFilter::Attrs attrs;
        attrs = attrs.DataFormat(is_nchw ? "NCHW" : "NHWC");
        attrs = attrs.Dilations({1, 1, 1, 1});
        
        // Create the operation
        auto depthwise_backprop_filter = tensorflow::ops::DepthwiseConv2dNativeBackpropFilter(
            root,
            input_placeholder,
            filter_sizes_placeholder,
            out_backprop_placeholder,
            {1, static_cast<int64_t>(stride_h), static_cast<int64_t>(stride_w), 1},
            use_same_padding ? "SAME" : "VALID",
            attrs
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run(
            {{input_placeholder, input_tensor},
             {filter_sizes_placeholder, filter_sizes_tensor},
             {out_backprop_placeholder, out_backprop_tensor}},
            {depthwise_backprop_filter},
            &outputs
        );
        
        if (!status.ok()) {
            std::cout << "Operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output shape
        if (!outputs.empty()) {
            const auto& output_tensor = outputs[0];
            tensorflow::TensorShape expected_filter_shape({filter_height, filter_width, in_channels, depth_multiplier});
            if (output_tensor.shape() != expected_filter_shape) {
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