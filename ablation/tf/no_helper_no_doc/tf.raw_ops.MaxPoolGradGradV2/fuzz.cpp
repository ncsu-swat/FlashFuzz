#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/kernel_def_builder.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions and parameters from fuzz input
        int batch_size = (data[offset] % 4) + 1;
        offset++;
        int height = (data[offset] % 8) + 2;
        offset++;
        int width = (data[offset] % 8) + 2;
        offset++;
        int channels = (data[offset] % 4) + 1;
        offset++;
        
        int ksize_h = (data[offset] % 3) + 1;
        offset++;
        int ksize_w = (data[offset] % 3) + 1;
        offset++;
        
        int stride_h = (data[offset] % 2) + 1;
        offset++;
        int stride_w = (data[offset] % 2) + 1;
        offset++;
        
        // Calculate output dimensions for max pool
        int out_height = (height - ksize_h) / stride_h + 1;
        int out_width = (width - ksize_w) / stride_w + 1;
        
        if (out_height <= 0 || out_width <= 0) return 0;
        
        // Create input tensors
        tensorflow::TensorShape orig_input_shape({batch_size, height, width, channels});
        tensorflow::TensorShape orig_output_shape({batch_size, out_height, out_width, channels});
        tensorflow::TensorShape grad_shape({batch_size, out_height, out_width, channels});
        
        tensorflow::Tensor orig_input(tensorflow::DT_FLOAT, orig_input_shape);
        tensorflow::Tensor orig_output(tensorflow::DT_FLOAT, orig_output_shape);
        tensorflow::Tensor grad(tensorflow::DT_FLOAT, grad_shape);
        
        // Fill tensors with fuzz data
        auto orig_input_flat = orig_input.flat<float>();
        auto orig_output_flat = orig_output.flat<float>();
        auto grad_flat = grad.flat<float>();
        
        size_t total_elements = orig_input_flat.size() + orig_output_flat.size() + grad_flat.size();
        size_t remaining_data = size - offset;
        
        for (int i = 0; i < orig_input_flat.size() && offset < size; i++) {
            orig_input_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
            offset++;
        }
        
        for (int i = 0; i < orig_output_flat.size() && offset < size; i++) {
            orig_output_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
            offset++;
        }
        
        for (int i = 0; i < grad_flat.size() && offset < size; i++) {
            grad_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
            offset++;
        }
        
        // Create session and graph
        tensorflow::GraphDefBuilder builder(tensorflow::GraphDefBuilder::kFailImmediately);
        
        // Create placeholder nodes
        auto orig_input_node = tensorflow::ops::Placeholder(
            builder.opts().WithName("orig_input").WithAttr("dtype", tensorflow::DT_FLOAT));
        auto orig_output_node = tensorflow::ops::Placeholder(
            builder.opts().WithName("orig_output").WithAttr("dtype", tensorflow::DT_FLOAT));
        auto grad_node = tensorflow::ops::Placeholder(
            builder.opts().WithName("grad").WithAttr("dtype", tensorflow::DT_FLOAT));
        
        // Create ksize and strides tensors
        std::vector<int32> ksize_vec = {1, ksize_h, ksize_w, 1};
        std::vector<int32> strides_vec = {1, stride_h, stride_w, 1};
        
        tensorflow::Tensor ksize_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({4}));
        tensorflow::Tensor strides_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({4}));
        
        auto ksize_flat = ksize_tensor.flat<int32>();
        auto strides_flat = strides_tensor.flat<int32>();
        
        for (int i = 0; i < 4; i++) {
            ksize_flat(i) = ksize_vec[i];
            strides_flat(i) = strides_vec[i];
        }
        
        auto ksize_node = tensorflow::ops::Const(ksize_tensor, builder.opts().WithName("ksize"));
        auto strides_node = tensorflow::ops::Const(strides_tensor, builder.opts().WithName("strides"));
        
        // Create MaxPoolGradGradV2 operation
        auto maxpool_grad_grad = tensorflow::ops::MaxPoolGradGradV2(
            orig_input_node, orig_output_node, grad_node, ksize_node, strides_node,
            builder.opts().WithName("maxpool_grad_grad").WithAttr("padding", "VALID"));
        
        tensorflow::GraphDef graph_def;
        auto status = builder.ToGraphDef(&graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Create session
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the operation
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"orig_input:0", orig_input},
            {"orig_output:0", orig_output},
            {"grad:0", grad}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"maxpool_grad_grad:0"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Verify output shape is reasonable
            auto output_shape = outputs[0].shape();
            if (output_shape.dims() == 4 && 
                output_shape.dim_size(0) == batch_size &&
                output_shape.dim_size(1) == height &&
                output_shape.dim_size(2) == width &&
                output_shape.dim_size(3) == channels) {
                // Success - output has expected shape
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