#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/nn_ops.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions from fuzz data
        int batch = (data[offset] % 4) + 1; offset++;
        int in_height = (data[offset] % 16) + 1; offset++;
        int in_width = (data[offset] % 16) + 1; offset++;
        int in_channels = (data[offset] % 8) + 1; offset++;
        
        int filter_height = (data[offset] % 8) + 1; offset++;
        int filter_width = (data[offset] % 8) + 1; offset++;
        int out_channels = (data[offset] % 8) + 1; offset++;
        
        int stride_h = (data[offset] % 4) + 1; offset++;
        int stride_w = (data[offset] % 4) + 1; offset++;
        
        // Calculate output dimensions
        int out_height = (in_height - filter_height) / stride_h + 1;
        int out_width = (in_width - filter_width) / stride_w + 1;
        
        if (out_height <= 0 || out_width <= 0) return 0;
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create input tensor (activations)
        tensorflow::TensorShape input_shape({batch, in_height, in_width, in_channels});
        auto input_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT, 
            tensorflow::ops::Placeholder::Shape(input_shape));
        
        // Create filter size tensor
        tensorflow::TensorShape filter_size_shape({4});
        auto filter_size_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32,
            tensorflow::ops::Placeholder::Shape(filter_size_shape));
        
        // Create out_backprop tensor (gradient w.r.t. output)
        tensorflow::TensorShape out_backprop_shape({batch, out_height, out_width, out_channels});
        auto out_backprop_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT,
            tensorflow::ops::Placeholder::Shape(out_backprop_shape));
        
        // Create Conv2DBackpropFilter operation
        auto conv2d_backprop_filter = tensorflow::ops::Conv2DBackpropFilter(
            root, input_placeholder, filter_size_placeholder, out_backprop_placeholder,
            {1, stride_h, stride_w, 1}, "VALID");
        
        // Create session
        tensorflow::ClientSession session(root);
        
        // Create input tensors with fuzz data
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, input_shape);
        auto input_flat = input_tensor.flat<float>();
        for (int i = 0; i < input_flat.size() && offset < size; ++i) {
            input_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
            offset++;
        }
        
        tensorflow::Tensor filter_size_tensor(tensorflow::DT_INT32, filter_size_shape);
        auto filter_size_flat = filter_size_tensor.flat<int32_t>();
        filter_size_flat(0) = filter_height;
        filter_size_flat(1) = filter_width;
        filter_size_flat(2) = in_channels;
        filter_size_flat(3) = out_channels;
        
        tensorflow::Tensor out_backprop_tensor(tensorflow::DT_FLOAT, out_backprop_shape);
        auto out_backprop_flat = out_backprop_tensor.flat<float>();
        for (int i = 0; i < out_backprop_flat.size() && offset < size; ++i) {
            out_backprop_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
            offset++;
        }
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status status = session.Run({
            {input_placeholder, input_tensor},
            {filter_size_placeholder, filter_size_tensor},
            {out_backprop_placeholder, out_backprop_tensor}
        }, {conv2d_backprop_filter}, &outputs);
        
        if (!status.ok()) {
            std::cout << "Conv2DBackpropFilter failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify output shape
        if (!outputs.empty()) {
            const tensorflow::Tensor& result = outputs[0];
            tensorflow::TensorShape expected_shape({filter_height, filter_width, in_channels, out_channels});
            if (result.shape() != expected_shape) {
                std::cout << "Unexpected output shape" << std::endl;
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