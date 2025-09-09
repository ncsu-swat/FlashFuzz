#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/const_op.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions for input tensor (4D: batch, height, width, channels)
        int32_t batch = (data[offset] % 4) + 1;
        offset++;
        int32_t height = (data[offset] % 32) + 1;
        offset++;
        int32_t width = (data[offset] % 32) + 1;
        offset++;
        int32_t channels = (data[offset] % 16) + 1;
        offset++;
        
        // Extract ksize parameters
        int32_t ksize_h = (data[offset] % 8) + 1;
        offset++;
        int32_t ksize_w = (data[offset] % 8) + 1;
        offset++;
        
        // Extract stride parameters
        int32_t stride_h = (data[offset] % 4) + 1;
        offset++;
        int32_t stride_w = (data[offset] % 4) + 1;
        offset++;
        
        // Extract padding type
        bool use_same_padding = (data[offset] % 2) == 0;
        offset++;
        
        // Extract data format
        int data_format_idx = data[offset] % 3;
        offset++;
        
        std::string padding = use_same_padding ? "SAME" : "VALID";
        std::string data_format;
        tensorflow::TensorShape input_shape;
        
        switch (data_format_idx) {
            case 0:
                data_format = "NHWC";
                input_shape = tensorflow::TensorShape({batch, height, width, channels});
                break;
            case 1:
                data_format = "NCHW";
                input_shape = tensorflow::TensorShape({batch, channels, height, width});
                break;
            case 2:
                data_format = "NCHW_VECT_C";
                // For NCHW_VECT_C, channels must be divisible by 4
                channels = ((channels + 3) / 4) * 4;
                input_shape = tensorflow::TensorShape({batch, channels/4, height, width, 4});
                break;
        }
        
        // Create TensorFlow scope
        auto root = tensorflow::Scope::NewRootScope();
        
        // Create input tensor
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, input_shape);
        auto input_flat = input_tensor.flat<float>();
        
        // Fill input tensor with fuzz data
        size_t tensor_size = input_tensor.NumElements();
        for (size_t i = 0; i < tensor_size && offset < size; ++i) {
            input_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
            offset++;
        }
        
        // Create ksize tensor
        tensorflow::Tensor ksize_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({4}));
        auto ksize_flat = ksize_tensor.flat<int32_t>();
        if (data_format == "NHWC") {
            ksize_flat(0) = 1;  // batch
            ksize_flat(1) = ksize_h;  // height
            ksize_flat(2) = ksize_w;  // width
            ksize_flat(3) = 1;  // channels
        } else {
            ksize_flat(0) = 1;  // batch
            ksize_flat(1) = 1;  // channels
            ksize_flat(2) = ksize_h;  // height
            ksize_flat(3) = ksize_w;  // width
        }
        
        // Create strides tensor
        tensorflow::Tensor strides_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({4}));
        auto strides_flat = strides_tensor.flat<int32_t>();
        if (data_format == "NHWC") {
            strides_flat(0) = 1;  // batch
            strides_flat(1) = stride_h;  // height
            strides_flat(2) = stride_w;  // width
            strides_flat(3) = 1;  // channels
        } else {
            strides_flat(0) = 1;  // batch
            strides_flat(1) = 1;  // channels
            strides_flat(2) = stride_h;  // height
            strides_flat(3) = stride_w;  // width
        }
        
        // Create constant ops
        auto input_op = tensorflow::ops::Const(root, input_tensor);
        auto ksize_op = tensorflow::ops::Const(root, ksize_tensor);
        auto strides_op = tensorflow::ops::Const(root, strides_tensor);
        
        // Create MaxPoolV2 operation
        auto maxpool_op = tensorflow::ops::MaxPoolV2(
            root,
            input_op,
            ksize_op,
            strides_op,
            padding,
            tensorflow::ops::MaxPoolV2::DataFormat(data_format)
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        auto status = session.Run({maxpool_op}, &outputs);
        
        if (!status.ok()) {
            // Operation failed, but this is expected for some invalid inputs
            return 0;
        }
        
        // Verify output tensor properties
        if (!outputs.empty()) {
            const auto& output = outputs[0];
            if (output.dtype() != tensorflow::DT_FLOAT) {
                return -1;
            }
            
            // Basic sanity check on output dimensions
            if (output.dims() != input_tensor.dims()) {
                return -1;
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}