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
        
        // Extract dimensions from fuzz data
        int batch = (data[offset++] % 4) + 1;
        int depth = (data[offset++] % 4) + 1;
        int rows = (data[offset++] % 4) + 1;
        int cols = (data[offset++] % 4) + 1;
        int in_channels = (data[offset++] % 4) + 1;
        int out_channels = (data[offset++] % 4) + 1;
        
        // Extract stride parameters
        int stride_d = (data[offset++] % 3) + 1;
        int stride_r = (data[offset++] % 3) + 1;
        int stride_c = (data[offset++] % 3) + 1;
        
        // Extract dilation parameters
        int dilation_d = (data[offset++] % 3) + 1;
        int dilation_r = (data[offset++] % 3) + 1;
        int dilation_c = (data[offset++] % 3) + 1;
        
        // Extract padding type
        bool use_same_padding = (data[offset++] % 2) == 0;
        
        // Calculate output dimensions based on padding
        int out_depth, out_rows, out_cols;
        if (use_same_padding) {
            out_depth = (depth + stride_d - 1) / stride_d;
            out_rows = (rows + stride_r - 1) / stride_r;
            out_cols = (cols + stride_c - 1) / stride_c;
        } else {
            out_depth = (depth - dilation_d * (depth - 1) + stride_d - 1) / stride_d;
            out_rows = (rows - dilation_r * (rows - 1) + stride_r - 1) / stride_r;
            out_cols = (cols - dilation_c * (cols - 1) + stride_c - 1) / stride_c;
            if (out_depth <= 0) out_depth = 1;
            if (out_rows <= 0) out_rows = 1;
            if (out_cols <= 0) out_cols = 1;
        }
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input tensor shape [batch, depth, rows, cols, in_channels]
        tensorflow::TensorShape input_shape({batch, depth, rows, cols, in_channels});
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, input_shape);
        auto input_flat = input_tensor.flat<float>();
        
        // Fill input tensor with fuzz data
        size_t input_size = input_flat.size();
        for (size_t i = 0; i < input_size && offset < size; ++i) {
            input_flat(i) = static_cast<float>(data[offset++]) / 255.0f;
        }
        
        // Create filter tensor shape [depth, rows, cols, in_channels, out_channels]
        tensorflow::TensorShape filter_shape({depth, rows, cols, in_channels, out_channels});
        tensorflow::Tensor filter_tensor(tensorflow::DT_FLOAT, filter_shape);
        auto filter_flat = filter_tensor.flat<float>();
        
        // Fill filter tensor with fuzz data
        size_t filter_size = filter_flat.size();
        for (size_t i = 0; i < filter_size && offset < size; ++i) {
            filter_flat(i) = static_cast<float>(data[offset++]) / 255.0f;
        }
        
        // Create out_backprop tensor shape [batch, out_depth, out_rows, out_cols, out_channels]
        tensorflow::TensorShape out_backprop_shape({batch, out_depth, out_rows, out_cols, out_channels});
        tensorflow::Tensor out_backprop_tensor(tensorflow::DT_FLOAT, out_backprop_shape);
        auto out_backprop_flat = out_backprop_tensor.flat<float>();
        
        // Fill out_backprop tensor with fuzz data
        size_t out_backprop_size = out_backprop_flat.size();
        for (size_t i = 0; i < out_backprop_size && offset < size; ++i) {
            out_backprop_flat(i) = static_cast<float>(data[offset++]) / 255.0f;
        }
        
        // Create input placeholders
        auto input_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto filter_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto out_backprop_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        // Set up strides (must have strides[0] = strides[4] = 1)
        std::vector<int> strides = {1, stride_d, stride_r, stride_c, 1};
        
        // Set up dilations
        std::vector<int> dilations = {1, dilation_d, dilation_r, dilation_c, 1};
        
        // Set up padding
        std::string padding = use_same_padding ? "SAME" : "VALID";
        
        // Create Conv3DBackpropInput operation
        auto conv3d_backprop_input = tensorflow::ops::Conv3DBackpropInput(
            root,
            tensorflow::ops::Const(root, input_shape.dim_sizes()),
            filter_ph,
            out_backprop_ph,
            strides,
            padding,
            tensorflow::ops::Conv3DBackpropInput::Dilations(dilations)
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run(
            {{filter_ph, filter_tensor}, {out_backprop_ph, out_backprop_tensor}},
            {conv3d_backprop_input},
            &outputs
        );
        
        if (!status.ok()) {
            std::cout << "TensorFlow operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output shape matches input shape
        if (!outputs.empty()) {
            const auto& output_shape = outputs[0].shape();
            if (output_shape.dims() == input_shape.dims()) {
                for (int i = 0; i < output_shape.dims(); ++i) {
                    if (output_shape.dim_size(i) != input_shape.dim_size(i)) {
                        std::cout << "Output shape mismatch at dimension " << i << std::endl;
                    }
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