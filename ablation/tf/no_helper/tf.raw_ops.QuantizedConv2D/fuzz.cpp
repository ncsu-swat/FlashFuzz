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
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/const_op.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 64) return 0;
        
        // Extract dimensions and parameters from fuzz data
        int batch = (data[offset] % 4) + 1; offset++;
        int height = (data[offset] % 32) + 1; offset++;
        int width = (data[offset] % 32) + 1; offset++;
        int input_depth = (data[offset] % 16) + 1; offset++;
        int filter_height = (data[offset] % 8) + 1; offset++;
        int filter_width = (data[offset] % 8) + 1; offset++;
        int output_depth = (data[offset] % 16) + 1; offset++;
        
        // Extract strides
        int stride_h = (data[offset] % 4) + 1; offset++;
        int stride_w = (data[offset] % 4) + 1; offset++;
        
        // Extract padding type
        bool use_same_padding = data[offset] % 2; offset++;
        
        // Extract quantization parameters
        float min_input = -10.0f + (data[offset] % 100) / 10.0f; offset++;
        float max_input = 1.0f + (data[offset] % 100) / 10.0f; offset++;
        float min_filter = -5.0f + (data[offset] % 50) / 10.0f; offset++;
        float max_filter = 1.0f + (data[offset] % 50) / 10.0f; offset++;
        
        // Extract dilations
        int dilation_h = (data[offset] % 3) + 1; offset++;
        int dilation_w = (data[offset] % 3) + 1; offset++;
        
        // Extract output type
        int out_type_idx = data[offset] % 5; offset++;
        tensorflow::DataType out_type;
        switch(out_type_idx) {
            case 0: out_type = tensorflow::DT_QINT8; break;
            case 1: out_type = tensorflow::DT_QUINT8; break;
            case 2: out_type = tensorflow::DT_QINT32; break;
            case 3: out_type = tensorflow::DT_QINT16; break;
            case 4: out_type = tensorflow::DT_QUINT16; break;
            default: out_type = tensorflow::DT_QINT32; break;
        }
        
        // Create TensorFlow scope
        auto root = tensorflow::Scope::NewRootScope();
        
        // Create input tensor (NHWC format)
        tensorflow::TensorShape input_shape({batch, height, width, input_depth});
        auto input_tensor = tensorflow::Tensor(tensorflow::DT_QUINT8, input_shape);
        auto input_flat = input_tensor.flat<tensorflow::quint8>();
        
        // Fill input with fuzz data
        for (int i = 0; i < input_flat.size() && offset < size; i++, offset++) {
            input_flat(i) = tensorflow::quint8(data[offset % size]);
        }
        
        // Create filter tensor (HWIO format)
        tensorflow::TensorShape filter_shape({filter_height, filter_width, input_depth, output_depth});
        auto filter_tensor = tensorflow::Tensor(tensorflow::DT_QUINT8, filter_shape);
        auto filter_flat = filter_tensor.flat<tensorflow::quint8>();
        
        // Fill filter with fuzz data
        for (int i = 0; i < filter_flat.size() && offset < size; i++, offset++) {
            filter_flat(i) = tensorflow::quint8(data[offset % size]);
        }
        
        // Create scalar tensors for min/max values
        auto min_input_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        min_input_tensor.scalar<float>()() = min_input;
        
        auto max_input_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        max_input_tensor.scalar<float>()() = max_input;
        
        auto min_filter_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        min_filter_tensor.scalar<float>()() = min_filter;
        
        auto max_filter_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        max_filter_tensor.scalar<float>()() = max_filter;
        
        // Create constant ops
        auto input_op = tensorflow::ops::Const(root, input_tensor);
        auto filter_op = tensorflow::ops::Const(root, filter_tensor);
        auto min_input_op = tensorflow::ops::Const(root, min_input_tensor);
        auto max_input_op = tensorflow::ops::Const(root, max_input_tensor);
        auto min_filter_op = tensorflow::ops::Const(root, min_filter_tensor);
        auto max_filter_op = tensorflow::ops::Const(root, max_filter_tensor);
        
        // Set up operation attributes
        tensorflow::ops::QuantizedConv2D::Attrs attrs;
        attrs = attrs.OutType(out_type);
        attrs = attrs.Dilations({1, dilation_h, dilation_w, 1});
        
        std::string padding_str = use_same_padding ? "SAME" : "VALID";
        std::vector<int> strides = {1, stride_h, stride_w, 1};
        
        // Create QuantizedConv2D operation
        auto quantized_conv2d = tensorflow::ops::QuantizedConv2D(
            root,
            input_op,
            filter_op,
            min_input_op,
            max_input_op,
            min_filter_op,
            max_filter_op,
            strides,
            padding_str,
            attrs
        );
        
        // Create session and run
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        auto status = session.Run({quantized_conv2d.output, quantized_conv2d.min_output, quantized_conv2d.max_output}, &outputs);
        
        if (!status.ok()) {
            std::cout << "QuantizedConv2D operation failed: " << status.ToString() << std::endl;
            return 0;
        }
        
        // Verify outputs
        if (outputs.size() != 3) {
            std::cout << "Expected 3 outputs, got " << outputs.size() << std::endl;
            return 0;
        }
        
        // Basic validation of output shapes and types
        if (outputs[0].dtype() != out_type) {
            std::cout << "Output type mismatch" << std::endl;
            return 0;
        }
        
        if (outputs[1].dtype() != tensorflow::DT_FLOAT || outputs[2].dtype() != tensorflow::DT_FLOAT) {
            std::cout << "Min/max output type should be float32" << std::endl;
            return 0;
        }
        
        // Check that output tensor has reasonable dimensions
        auto output_shape = outputs[0].shape();
        if (output_shape.dims() != 4) {
            std::cout << "Output should be 4D tensor" << std::endl;
            return 0;
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}