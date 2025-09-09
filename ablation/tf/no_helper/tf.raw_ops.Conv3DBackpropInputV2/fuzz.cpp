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
        int32_t batch = (data[offset] % 4) + 1; offset++;
        int32_t in_depth = (data[offset] % 8) + 1; offset++;
        int32_t in_height = (data[offset] % 8) + 1; offset++;
        int32_t in_width = (data[offset] % 8) + 1; offset++;
        int32_t in_channels = (data[offset] % 8) + 1; offset++;
        int32_t out_channels = (data[offset] % 8) + 1; offset++;
        
        int32_t filter_depth = (data[offset] % 5) + 1; offset++;
        int32_t filter_height = (data[offset] % 5) + 1; offset++;
        int32_t filter_width = (data[offset] % 5) + 1; offset++;
        
        // Calculate output dimensions based on padding
        bool use_same_padding = (data[offset] % 2) == 0; offset++;
        std::string padding = use_same_padding ? "SAME" : "VALID";
        
        int32_t stride_d = (data[offset] % 3) + 1; offset++;
        int32_t stride_h = (data[offset] % 3) + 1; offset++;
        int32_t stride_w = (data[offset] % 3) + 1; offset++;
        
        int32_t out_depth, out_height, out_width;
        if (use_same_padding) {
            out_depth = (in_depth + stride_d - 1) / stride_d;
            out_height = (in_height + stride_h - 1) / stride_h;
            out_width = (in_width + stride_w - 1) / stride_w;
        } else {
            out_depth = (in_depth - filter_depth) / stride_d + 1;
            out_height = (in_height - filter_height) / stride_h + 1;
            out_width = (in_width - filter_width) / stride_w + 1;
        }
        
        if (out_depth <= 0 || out_height <= 0 || out_width <= 0) return 0;
        
        bool use_ndhwc = (data[offset] % 2) == 0; offset++;
        std::string data_format = use_ndhwc ? "NDHWC" : "NCDHW";
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input_sizes tensor
        tensorflow::Tensor input_sizes_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({5}));
        auto input_sizes_flat = input_sizes_tensor.flat<int32_t>();
        input_sizes_flat(0) = batch;
        input_sizes_flat(1) = in_depth;
        input_sizes_flat(2) = in_height;
        input_sizes_flat(3) = in_width;
        input_sizes_flat(4) = in_channels;
        
        auto input_sizes = tensorflow::ops::Const(root, input_sizes_tensor);
        
        // Create filter tensor
        tensorflow::TensorShape filter_shape({filter_depth, filter_height, filter_width, in_channels, out_channels});
        tensorflow::Tensor filter_tensor(tensorflow::DT_FLOAT, filter_shape);
        auto filter_flat = filter_tensor.flat<float>();
        
        size_t filter_size = filter_flat.size();
        for (size_t i = 0; i < filter_size && offset < size; ++i) {
            filter_flat(i) = static_cast<float>(data[offset % size]) / 255.0f - 0.5f;
            offset++;
        }
        
        auto filter = tensorflow::ops::Const(root, filter_tensor);
        
        // Create out_backprop tensor
        tensorflow::TensorShape out_backprop_shape({batch, out_depth, out_height, out_width, out_channels});
        tensorflow::Tensor out_backprop_tensor(tensorflow::DT_FLOAT, out_backprop_shape);
        auto out_backprop_flat = out_backprop_tensor.flat<float>();
        
        size_t out_backprop_size = out_backprop_flat.size();
        for (size_t i = 0; i < out_backprop_size && offset < size; ++i) {
            out_backprop_flat(i) = static_cast<float>(data[offset % size]) / 255.0f - 0.5f;
            offset++;
        }
        
        auto out_backprop = tensorflow::ops::Const(root, out_backprop_tensor);
        
        // Set up operation attributes
        std::vector<int> strides = {1, stride_d, stride_h, stride_w, 1};
        std::vector<int> dilations = {1, 1, 1, 1, 1};
        
        // Create Conv3DBackpropInputV2 operation
        auto conv3d_backprop = tensorflow::ops::Conv3DBackpropInputV2(
            root,
            input_sizes,
            filter,
            out_backprop,
            strides,
            padding,
            tensorflow::ops::Conv3DBackpropInputV2::DataFormat(data_format),
            tensorflow::ops::Conv3DBackpropInputV2::Dilations(dilations)
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({conv3d_backprop}, &outputs);
        
        if (!status.ok()) {
            std::cout << "Conv3DBackpropInputV2 failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output shape
        if (!outputs.empty()) {
            const tensorflow::Tensor& result = outputs[0];
            if (result.dims() != 5) {
                std::cout << "Unexpected output dimensions: " << result.dims() << std::endl;
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