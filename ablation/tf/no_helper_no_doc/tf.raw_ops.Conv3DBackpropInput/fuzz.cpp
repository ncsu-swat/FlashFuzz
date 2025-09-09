#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/framework/fake_input.h>
#include <tensorflow/core/kernels/ops_testutil.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 64) return 0;
        
        // Extract dimensions and parameters from fuzz input
        int32_t batch_size = (data[offset] % 4) + 1;
        offset++;
        int32_t input_depth = (data[offset] % 8) + 1;
        offset++;
        int32_t input_height = (data[offset] % 16) + 1;
        offset++;
        int32_t input_width = (data[offset] % 16) + 1;
        offset++;
        int32_t input_channels = (data[offset] % 8) + 1;
        offset++;
        
        int32_t filter_depth = (data[offset] % 5) + 1;
        offset++;
        int32_t filter_height = (data[offset] % 5) + 1;
        offset++;
        int32_t filter_width = (data[offset] % 5) + 1;
        offset++;
        int32_t out_channels = (data[offset] % 8) + 1;
        offset++;
        
        // Strides
        int32_t stride_d = (data[offset] % 3) + 1;
        offset++;
        int32_t stride_h = (data[offset] % 3) + 1;
        offset++;
        int32_t stride_w = (data[offset] % 3) + 1;
        offset++;
        
        // Padding type
        bool use_same_padding = data[offset] % 2;
        offset++;
        
        if (offset >= size) return 0;
        
        // Calculate output dimensions
        int32_t out_depth, out_height, out_width;
        if (use_same_padding) {
            out_depth = (input_depth + stride_d - 1) / stride_d;
            out_height = (input_height + stride_h - 1) / stride_h;
            out_width = (input_width + stride_w - 1) / stride_w;
        } else {
            out_depth = (input_depth - filter_depth) / stride_d + 1;
            out_height = (input_height - filter_height) / stride_h + 1;
            out_width = (input_width - filter_width) / stride_w + 1;
            if (out_depth <= 0 || out_height <= 0 || out_width <= 0) return 0;
        }
        
        // Create input_sizes tensor (shape of input to be reconstructed)
        tensorflow::Tensor input_sizes(tensorflow::DT_INT32, tensorflow::TensorShape({5}));
        auto input_sizes_flat = input_sizes.flat<int32_t>();
        input_sizes_flat(0) = batch_size;
        input_sizes_flat(1) = input_depth;
        input_sizes_flat(2) = input_height;
        input_sizes_flat(3) = input_width;
        input_sizes_flat(4) = input_channels;
        
        // Create filter tensor
        tensorflow::TensorShape filter_shape({filter_depth, filter_height, filter_width, input_channels, out_channels});
        tensorflow::Tensor filter(tensorflow::DT_FLOAT, filter_shape);
        auto filter_flat = filter.flat<float>();
        
        // Fill filter with fuzz data
        size_t filter_size = filter_flat.size();
        for (int i = 0; i < filter_size && offset < size; i++) {
            filter_flat(i) = static_cast<float>(data[offset]) / 255.0f - 0.5f;
            offset++;
        }
        
        // Create out_backprop tensor (gradient w.r.t. output)
        tensorflow::TensorShape out_backprop_shape({batch_size, out_depth, out_height, out_width, out_channels});
        tensorflow::Tensor out_backprop(tensorflow::DT_FLOAT, out_backprop_shape);
        auto out_backprop_flat = out_backprop.flat<float>();
        
        // Fill out_backprop with remaining fuzz data
        size_t backprop_size = out_backprop_flat.size();
        for (int i = 0; i < backprop_size && offset < size; i++) {
            out_backprop_flat(i) = static_cast<float>(data[offset]) / 255.0f - 0.5f;
            offset++;
        }
        
        // Create a simple test using OpsTestBase
        class Conv3DBackpropInputTest : public tensorflow::OpsTestBase {
        public:
            void RunTest(const tensorflow::Tensor& input_sizes,
                        const tensorflow::Tensor& filter,
                        const tensorflow::Tensor& out_backprop,
                        const std::string& padding) {
                
                tensorflow::NodeDefBuilder builder("conv3d_backprop_input", "Conv3DBackpropInput");
                builder.Input(tensorflow::FakeInput(tensorflow::DT_INT32))
                       .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))
                       .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))
                       .Attr("strides", {1, stride_d, stride_h, stride_w, 1})
                       .Attr("padding", padding)
                       .Attr("T", tensorflow::DT_FLOAT);
                
                tensorflow::Status status = tensorflow::NodeDefBuilder::Finalize(builder, node_def());
                if (!status.ok()) return;
                
                status = InitOp();
                if (!status.ok()) return;
                
                AddInputFromArray<int32_t>(input_sizes.shape(), input_sizes.flat<int32_t>());
                AddInputFromArray<float>(filter.shape(), filter.flat<float>());
                AddInputFromArray<float>(out_backprop.shape(), out_backprop.flat<float>());
                
                status = RunOpKernel();
                // Don't check status as we expect some operations to fail with invalid inputs
            }
            
        private:
            int32_t stride_d, stride_h, stride_w;
        public:
            void SetStrides(int32_t d, int32_t h, int32_t w) {
                stride_d = d; stride_h = h; stride_w = w;
            }
        };
        
        Conv3DBackpropInputTest test;
        test.SetStrides(stride_d, stride_h, stride_w);
        
        std::string padding = use_same_padding ? "SAME" : "VALID";
        test.RunTest(input_sizes, filter, out_backprop, padding);
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}