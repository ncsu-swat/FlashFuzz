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
        
        if (size < 50) return 0;
        
        // Extract dimensions from fuzz data
        int batch = (data[offset++] % 4) + 1;
        int in_depth = (data[offset++] % 8) + 1;
        int in_height = (data[offset++] % 8) + 1;
        int in_width = (data[offset++] % 8) + 1;
        int in_channels = (data[offset++] % 4) + 1;
        
        int filter_depth = (data[offset++] % 4) + 1;
        int filter_height = (data[offset++] % 4) + 1;
        int filter_width = (data[offset++] % 4) + 1;
        int out_channels = (data[offset++] % 4) + 1;
        
        // Calculate output dimensions based on VALID padding
        int out_depth = in_depth - filter_depth + 1;
        int out_height = in_height - filter_height + 1;
        int out_width = in_width - filter_width + 1;
        
        if (out_depth <= 0 || out_height <= 0 || out_width <= 0) return 0;
        
        // Extract strides (must have strides[0] = strides[4] = 1)
        int stride_depth = (data[offset++] % 3) + 1;
        int stride_height = (data[offset++] % 3) + 1;
        int stride_width = (data[offset++] % 3) + 1;
        
        // Extract padding type
        bool use_same_padding = (data[offset++] % 2) == 0;
        
        // Extract data format
        bool use_ndhwc = (data[offset++] % 2) == 0;
        
        // Extract dilations (must have dilations[0] = dilations[4] = 1)
        int dilation_depth = (data[offset++] % 3) + 1;
        int dilation_height = (data[offset++] % 3) + 1;
        int dilation_width = (data[offset++] % 3) + 1;
        
        // Create TensorFlow session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create input tensor
        tensorflow::Tensor input_tensor;
        if (use_ndhwc) {
            input_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, 
                tensorflow::TensorShape({batch, in_depth, in_height, in_width, in_channels}));
        } else {
            input_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT,
                tensorflow::TensorShape({batch, in_channels, in_depth, in_height, in_width}));
        }
        
        // Fill input tensor with fuzz data
        auto input_flat = input_tensor.flat<float>();
        for (int i = 0; i < input_flat.size() && offset < size; ++i) {
            input_flat(i) = static_cast<float>(data[offset++]) / 255.0f;
        }
        
        // Create filter_sizes tensor
        tensorflow::Tensor filter_sizes_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({5}));
        auto filter_sizes_flat = filter_sizes_tensor.flat<int32_t>();
        filter_sizes_flat(0) = filter_depth;
        filter_sizes_flat(1) = filter_height;
        filter_sizes_flat(2) = filter_width;
        filter_sizes_flat(3) = in_channels;
        filter_sizes_flat(4) = out_channels;
        
        // Recalculate output dimensions with padding and strides
        if (use_same_padding) {
            out_depth = (in_depth + stride_depth - 1) / stride_depth;
            out_height = (in_height + stride_height - 1) / stride_height;
            out_width = (in_width + stride_width - 1) / stride_width;
        } else {
            out_depth = (in_depth - (filter_depth - 1) * dilation_depth + stride_depth - 1) / stride_depth;
            out_height = (in_height - (filter_height - 1) * dilation_height + stride_height - 1) / stride_height;
            out_width = (in_width - (filter_width - 1) * dilation_width + stride_width - 1) / stride_width;
        }
        
        if (out_depth <= 0 || out_height <= 0 || out_width <= 0) return 0;
        
        // Create out_backprop tensor
        tensorflow::Tensor out_backprop_tensor;
        if (use_ndhwc) {
            out_backprop_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT,
                tensorflow::TensorShape({batch, out_depth, out_height, out_width, out_channels}));
        } else {
            out_backprop_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT,
                tensorflow::TensorShape({batch, out_channels, out_depth, out_height, out_width}));
        }
        
        // Fill out_backprop tensor with fuzz data
        auto out_backprop_flat = out_backprop_tensor.flat<float>();
        for (int i = 0; i < out_backprop_flat.size() && offset < size; ++i) {
            out_backprop_flat(i) = static_cast<float>(data[offset++]) / 255.0f;
        }
        
        // Create graph definition
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        node_def->set_name("conv3d_backprop_filter");
        node_def->set_op("Conv3DBackpropFilterV2");
        
        // Add input names
        node_def->add_input("input:0");
        node_def->add_input("filter_sizes:0");
        node_def->add_input("out_backprop:0");
        
        // Set attributes
        tensorflow::AttrValue strides_attr;
        strides_attr.mutable_list()->add_i(1);
        strides_attr.mutable_list()->add_i(stride_depth);
        strides_attr.mutable_list()->add_i(stride_height);
        strides_attr.mutable_list()->add_i(stride_width);
        strides_attr.mutable_list()->add_i(1);
        (*node_def->mutable_attr())["strides"] = strides_attr;
        
        tensorflow::AttrValue padding_attr;
        padding_attr.set_s(use_same_padding ? "SAME" : "VALID");
        (*node_def->mutable_attr())["padding"] = padding_attr;
        
        tensorflow::AttrValue data_format_attr;
        data_format_attr.set_s(use_ndhwc ? "NDHWC" : "NCDHW");
        (*node_def->mutable_attr())["data_format"] = data_format_attr;
        
        tensorflow::AttrValue dilations_attr;
        dilations_attr.mutable_list()->add_i(1);
        dilations_attr.mutable_list()->add_i(dilation_depth);
        dilations_attr.mutable_list()->add_i(dilation_height);
        dilations_attr.mutable_list()->add_i(dilation_width);
        dilations_attr.mutable_list()->add_i(1);
        (*node_def->mutable_attr())["dilations"] = dilations_attr;
        
        tensorflow::AttrValue type_attr;
        type_attr.set_type(tensorflow::DT_FLOAT);
        (*node_def->mutable_attr())["T"] = type_attr;
        
        // Add input nodes
        tensorflow::NodeDef* input_node = graph_def.add_node();
        input_node->set_name("input");
        input_node->set_op("Placeholder");
        tensorflow::AttrValue input_dtype_attr;
        input_dtype_attr.set_type(tensorflow::DT_FLOAT);
        (*input_node->mutable_attr())["dtype"] = input_dtype_attr;
        
        tensorflow::NodeDef* filter_sizes_node = graph_def.add_node();
        filter_sizes_node->set_name("filter_sizes");
        filter_sizes_node->set_op("Placeholder");
        tensorflow::AttrValue filter_sizes_dtype_attr;
        filter_sizes_dtype_attr.set_type(tensorflow::DT_INT32);
        (*filter_sizes_node->mutable_attr())["dtype"] = filter_sizes_dtype_attr;
        
        tensorflow::NodeDef* out_backprop_node = graph_def.add_node();
        out_backprop_node->set_name("out_backprop");
        out_backprop_node->set_op("Placeholder");
        tensorflow::AttrValue out_backprop_dtype_attr;
        out_backprop_dtype_attr.set_type(tensorflow::DT_FLOAT);
        (*out_backprop_node->mutable_attr())["dtype"] = out_backprop_dtype_attr;
        
        // Create session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) return 0;
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"input:0", input_tensor},
            {"filter_sizes:0", filter_sizes_tensor},
            {"out_backprop:0", out_backprop_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {"conv3d_backprop_filter:0"};
        
        status = session->Run(inputs, output_names, {}, &outputs);
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}