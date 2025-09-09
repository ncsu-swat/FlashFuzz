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
#include <tensorflow/core/kernels/ops_util.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 64) return 0;
        
        // Extract dimensions and parameters from fuzzer input
        int32_t input_batch = (data[offset] % 4) + 1; offset++;
        int32_t input_height = (data[offset] % 32) + 1; offset++;
        int32_t input_width = (data[offset] % 32) + 1; offset++;
        int32_t input_channels = (data[offset] % 8) + 1; offset++;
        
        int32_t size_height = (data[offset] % 64) + 1; offset++;
        int32_t size_width = (data[offset] % 64) + 1; offset++;
        
        int32_t filter_height = (data[offset] % 8) + 1; offset++;
        int32_t filter_width = (data[offset] % 8) + 1; offset++;
        int32_t filter_out_channels = (data[offset] % 8) + 1; offset++;
        
        int32_t pad_top = data[offset] % 4; offset++;
        int32_t pad_bottom = data[offset] % 4; offset++;
        int32_t pad_left = data[offset] % 4; offset++;
        int32_t pad_right = data[offset] % 4; offset++;
        
        int32_t stride_h = (data[offset] % 3) + 1; offset++;
        int32_t stride_w = (data[offset] % 3) + 1; offset++;
        
        bool resize_align_corners = data[offset] % 2; offset++;
        
        std::string padding = (data[offset] % 2) ? "SAME" : "VALID"; offset++;
        std::string data_format = (data[offset] % 2) ? "NHWC" : "NCHW"; offset++;
        
        // Create input tensor
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, 
            tensorflow::TensorShape({input_batch, input_height, input_width, input_channels}));
        auto input_flat = input_tensor.flat<float>();
        for (int i = 0; i < input_flat.size() && offset < size; ++i) {
            input_flat(i) = static_cast<float>(data[offset++]) / 255.0f;
        }
        
        // Create size tensor
        tensorflow::Tensor size_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({2}));
        auto size_flat = size_tensor.flat<int32_t>();
        size_flat(0) = size_height;
        size_flat(1) = size_width;
        
        // Create filter tensor
        tensorflow::Tensor filter_tensor(tensorflow::DT_FLOAT,
            tensorflow::TensorShape({filter_height, filter_width, input_channels, filter_out_channels}));
        auto filter_flat = filter_tensor.flat<float>();
        for (int i = 0; i < filter_flat.size() && offset < size; ++i) {
            filter_flat(i) = static_cast<float>(data[offset++]) / 255.0f - 0.5f;
        }
        
        // Create paddings tensor
        tensorflow::Tensor paddings_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({4, 2}));
        auto paddings_flat = paddings_tensor.flat<int32_t>();
        paddings_flat(0) = 0; paddings_flat(1) = 0; // batch padding
        paddings_flat(2) = pad_top; paddings_flat(3) = pad_bottom; // height padding
        paddings_flat(4) = pad_left; paddings_flat(5) = pad_right; // width padding
        paddings_flat(6) = 0; paddings_flat(7) = 0; // channel padding
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create graph
        tensorflow::GraphDef graph_def;
        tensorflow::GraphDefBuilder builder(tensorflow::GraphDefBuilder::kFailImmediately);
        
        auto input_node = tensorflow::ops::Placeholder(builder.opts()
            .WithName("input")
            .WithAttr("dtype", tensorflow::DT_FLOAT)
            .WithAttr("shape", tensorflow::TensorShape({input_batch, input_height, input_width, input_channels})));
            
        auto size_node = tensorflow::ops::Placeholder(builder.opts()
            .WithName("size")
            .WithAttr("dtype", tensorflow::DT_INT32)
            .WithAttr("shape", tensorflow::TensorShape({2})));
            
        auto filter_node = tensorflow::ops::Placeholder(builder.opts()
            .WithName("filter")
            .WithAttr("dtype", tensorflow::DT_FLOAT)
            .WithAttr("shape", tensorflow::TensorShape({filter_height, filter_width, input_channels, filter_out_channels})));
            
        auto paddings_node = tensorflow::ops::Placeholder(builder.opts()
            .WithName("paddings")
            .WithAttr("dtype", tensorflow::DT_INT32)
            .WithAttr("shape", tensorflow::TensorShape({4, 2})));
        
        auto fused_op = tensorflow::ops::UnaryOp("FusedResizeAndPadConv2D", 
            {input_node, size_node, paddings_node, filter_node},
            builder.opts()
                .WithName("fused_resize_pad_conv2d")
                .WithAttr("T", tensorflow::DT_FLOAT)
                .WithAttr("resize_align_corners", resize_align_corners)
                .WithAttr("mode", std::string("REFLECT"))
                .WithAttr("strides", std::vector<int32_t>({1, stride_h, stride_w, 1}))
                .WithAttr("padding", padding)
                .WithAttr("data_format", data_format));
        
        tensorflow::Status status = builder.ToGraphDef(&graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({
            {"input", input_tensor},
            {"size", size_tensor},
            {"filter", filter_tensor},
            {"paddings", paddings_tensor}
        }, {"fused_resize_pad_conv2d"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Operation succeeded, check output validity
            const auto& output = outputs[0];
            if (output.NumElements() > 0) {
                auto output_flat = output.flat<float>();
                for (int i = 0; i < std::min(10, static_cast<int>(output_flat.size())); ++i) {
                    volatile float val = output_flat(i);
                    (void)val; // Prevent optimization
                }
            }
        }
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}