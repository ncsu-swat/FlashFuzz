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
        int in_height = (data[offset] % 8) + 1; offset++;
        int in_width = (data[offset] % 8) + 1; offset++;
        int in_channels = (data[offset] % 4) + 1; offset++;
        
        int filter_height = (data[offset] % 4) + 1; offset++;
        int filter_width = (data[offset] % 4) + 1; offset++;
        int channel_multiplier = (data[offset] % 2) + 1; offset++;
        
        int out_height = (data[offset] % 8) + 1; offset++;
        int out_width = (data[offset] % 8) + 1; offset++;
        int out_channels = in_channels * channel_multiplier;
        
        // Extract strides and padding
        int stride_h = (data[offset] % 2) + 1; offset++;
        int stride_w = (data[offset] % 2) + 1; offset++;
        bool use_same_padding = data[offset] % 2; offset++;
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input tensor
        tensorflow::TensorShape input_shape({batch, in_height, in_width, in_channels});
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, input_shape);
        auto input_flat = input_tensor.flat<float>();
        
        // Fill input tensor with fuzz data
        for (int i = 0; i < input_flat.size() && offset < size - 4; i++) {
            float val;
            memcpy(&val, data + offset, sizeof(float));
            input_flat(i) = val;
            offset += sizeof(float);
        }
        
        // Create filter size tensor
        tensorflow::TensorShape filter_size_shape({4});
        tensorflow::Tensor filter_size_tensor(tensorflow::DT_INT32, filter_size_shape);
        auto filter_size_flat = filter_size_tensor.flat<int32_t>();
        filter_size_flat(0) = filter_height;
        filter_size_flat(1) = filter_width;
        filter_size_flat(2) = in_channels;
        filter_size_flat(3) = channel_multiplier;
        
        // Create out_backprop tensor
        tensorflow::TensorShape out_backprop_shape({batch, out_height, out_width, out_channels});
        tensorflow::Tensor out_backprop_tensor(tensorflow::DT_FLOAT, out_backprop_shape);
        auto out_backprop_flat = out_backprop_tensor.flat<float>();
        
        // Fill out_backprop tensor with remaining fuzz data
        for (int i = 0; i < out_backprop_flat.size() && offset < size - 4; i++) {
            float val;
            memcpy(&val, data + offset, sizeof(float));
            out_backprop_flat(i) = val;
            offset += sizeof(float);
        }
        
        // Create input placeholders
        auto input_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto filter_size_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto out_backprop_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        // Set up operation attributes
        std::vector<int> strides = {1, stride_h, stride_w, 1};
        std::string padding = use_same_padding ? "SAME" : "VALID";
        
        // Create DepthwiseConv2dNativeBackpropFilter operation
        auto depthwise_backprop_filter = tensorflow::ops::DepthwiseConv2dNativeBackpropFilter(
            root, input_ph, filter_size_ph, out_backprop_ph,
            tensorflow::ops::DepthwiseConv2dNativeBackpropFilter::Strides(strides)
                .Padding(padding)
                .DataFormat("NHWC")
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run(
            {{input_ph, input_tensor}, 
             {filter_size_ph, filter_size_tensor}, 
             {out_backprop_ph, out_backprop_tensor}},
            {depthwise_backprop_filter},
            &outputs
        );
        
        if (!status.ok()) {
            std::cout << "TensorFlow operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}