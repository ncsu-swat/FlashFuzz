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
        int out_channels = (data[offset] % 4) + 1; offset++;
        
        int stride_h = (data[offset] % 3) + 1; offset++;
        int stride_w = (data[offset] % 3) + 1; offset++;
        
        // Calculate output dimensions
        int out_height = (in_height - filter_height) / stride_h + 1;
        int out_width = (in_width - filter_width) / stride_w + 1;
        
        if (out_height <= 0 || out_width <= 0) return 0;
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input tensor (activations)
        tensorflow::TensorShape input_shape({batch, in_height, in_width, in_channels});
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, input_shape);
        auto input_flat = input_tensor.flat<float>();
        
        // Fill input tensor with fuzz data
        size_t input_size = batch * in_height * in_width * in_channels;
        for (size_t i = 0; i < input_size && offset < size; i++, offset++) {
            input_flat(i) = static_cast<float>(data[offset % size]) / 255.0f - 0.5f;
        }
        
        // Create filter size tensor
        tensorflow::TensorShape filter_size_shape({4});
        tensorflow::Tensor filter_size_tensor(tensorflow::DT_INT32, filter_size_shape);
        auto filter_size_flat = filter_size_tensor.flat<int32_t>();
        filter_size_flat(0) = filter_height;
        filter_size_flat(1) = filter_width;
        filter_size_flat(2) = in_channels;
        filter_size_flat(3) = out_channels;
        
        // Create out_backprop tensor (gradient w.r.t. output)
        tensorflow::TensorShape out_backprop_shape({batch, out_height, out_width, out_channels});
        tensorflow::Tensor out_backprop_tensor(tensorflow::DT_FLOAT, out_backprop_shape);
        auto out_backprop_flat = out_backprop_tensor.flat<float>();
        
        // Fill out_backprop tensor with fuzz data
        size_t out_backprop_size = batch * out_height * out_width * out_channels;
        for (size_t i = 0; i < out_backprop_size && offset < size; i++, offset++) {
            out_backprop_flat(i) = static_cast<float>(data[offset % size]) / 255.0f - 0.5f;
        }
        
        // Create input placeholders
        auto input_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        auto filter_size_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
        auto out_backprop_ph = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
        
        // Set up Conv2DBackpropFilter operation
        tensorflow::ops::Conv2DBackpropFilter::Attrs attrs;
        attrs = attrs.Strides({1, stride_h, stride_w, 1});
        attrs = attrs.Padding("VALID");
        attrs = attrs.DataFormat("NHWC");
        
        auto conv2d_backprop_filter = tensorflow::ops::Conv2DBackpropFilter(
            root, input_ph, filter_size_ph, out_backprop_ph, attrs);
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status status = session.Run(
            {{input_ph, input_tensor}, 
             {filter_size_ph, filter_size_tensor}, 
             {out_backprop_ph, out_backprop_tensor}},
            {conv2d_backprop_filter}, 
            &outputs);
        
        if (!status.ok()) {
            std::cout << "TensorFlow operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output shape
        if (!outputs.empty()) {
            const auto& output = outputs[0];
            if (output.dims() == 4 && 
                output.dim_size(0) == filter_height &&
                output.dim_size(1) == filter_width &&
                output.dim_size(2) == in_channels &&
                output.dim_size(3) == out_channels) {
                // Output shape is correct
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}