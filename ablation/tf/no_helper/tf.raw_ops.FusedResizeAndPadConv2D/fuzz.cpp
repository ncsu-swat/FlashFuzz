#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        // Need at least basic structure
        if (size < 64) return 0;
        
        // Extract dimensions and parameters from fuzz data
        uint32_t batch = (data[offset] % 4) + 1; offset++;
        uint32_t in_height = (data[offset] % 32) + 1; offset++;
        uint32_t in_width = (data[offset] % 32) + 1; offset++;
        uint32_t in_channels = (data[offset] % 8) + 1; offset++;
        
        uint32_t filter_height = (data[offset] % 8) + 1; offset++;
        uint32_t filter_width = (data[offset] % 8) + 1; offset++;
        uint32_t out_channels = (data[offset] % 8) + 1; offset++;
        
        uint32_t new_height = (data[offset] % 64) + 1; offset++;
        uint32_t new_width = (data[offset] % 64) + 1; offset++;
        
        // Extract padding values
        uint32_t pad_top = data[offset] % 8; offset++;
        uint32_t pad_bottom = data[offset] % 8; offset++;
        uint32_t pad_left = data[offset] % 8; offset++;
        uint32_t pad_right = data[offset] % 8; offset++;
        
        // Extract stride values
        uint32_t stride_h = (data[offset] % 4) + 1; offset++;
        uint32_t stride_w = (data[offset] % 4) + 1; offset++;
        
        // Extract mode and padding type
        bool mode_reflect = (data[offset] % 2) == 0; offset++;
        bool padding_same = (data[offset] % 2) == 0; offset++;
        bool resize_align_corners = (data[offset] % 2) == 0; offset++;
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input tensor
        tensorflow::TensorShape input_shape({batch, in_height, in_width, in_channels});
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, input_shape);
        auto input_flat = input_tensor.flat<float>();
        
        // Fill input with fuzz data
        for (int i = 0; i < input_flat.size() && offset < size; ++i) {
            input_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
            offset++;
        }
        
        // Create size tensor
        tensorflow::Tensor size_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({2}));
        auto size_flat = size_tensor.flat<int32_t>();
        size_flat(0) = new_height;
        size_flat(1) = new_width;
        
        // Create paddings tensor
        tensorflow::Tensor paddings_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({4, 2}));
        auto paddings_flat = paddings_tensor.flat<int32_t>();
        paddings_flat(0) = 0; paddings_flat(1) = 0; // batch padding
        paddings_flat(2) = pad_top; paddings_flat(3) = pad_bottom; // height padding
        paddings_flat(4) = pad_left; paddings_flat(5) = pad_right; // width padding
        paddings_flat(6) = 0; paddings_flat(7) = 0; // channel padding
        
        // Create filter tensor
        tensorflow::TensorShape filter_shape({filter_height, filter_width, in_channels, out_channels});
        tensorflow::Tensor filter_tensor(tensorflow::DT_FLOAT, filter_shape);
        auto filter_flat = filter_tensor.flat<float>();
        
        // Fill filter with fuzz data
        for (int i = 0; i < filter_flat.size() && offset < size; ++i) {
            filter_flat(i) = static_cast<float>(data[offset % size]) / 255.0f - 0.5f;
            offset++;
        }
        
        // Create input placeholders
        auto input_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto size_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto paddings_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto filter_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        // Set up operation attributes
        std::string mode_str = mode_reflect ? "REFLECT" : "SYMMETRIC";
        std::string padding_str = padding_same ? "SAME" : "VALID";
        std::vector<int> strides = {1, static_cast<int>(stride_h), static_cast<int>(stride_w), 1};
        
        // Create the FusedResizeAndPadConv2D operation
        auto fused_op = tensorflow::ops::FusedResizeAndPadConv2D(
            root,
            input_ph,
            size_ph,
            paddings_ph,
            filter_ph,
            mode_str,
            strides,
            padding_str,
            tensorflow::ops::FusedResizeAndPadConv2D::ResizeAlignCorners(resize_align_corners)
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run(
            {{input_ph, input_tensor}, 
             {size_ph, size_tensor}, 
             {paddings_ph, paddings_tensor}, 
             {filter_ph, filter_tensor}},
            {fused_op},
            &outputs
        );
        
        // Check if operation succeeded
        if (!status.ok()) {
            // Operation failed, but this is expected for some invalid inputs
            return 0;
        }
        
        // Verify output tensor properties
        if (!outputs.empty()) {
            const auto& output = outputs[0];
            if (output.dims() == 4 && output.dim_size(0) == batch) {
                // Basic validation passed
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}