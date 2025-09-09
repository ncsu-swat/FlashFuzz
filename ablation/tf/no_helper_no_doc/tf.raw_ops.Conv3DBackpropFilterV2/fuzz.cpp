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
        
        // Extract dimensions from fuzz data
        int batch_size = (data[offset] % 4) + 1;
        offset++;
        int input_depth = (data[offset] % 8) + 1;
        offset++;
        int input_height = (data[offset] % 8) + 1;
        offset++;
        int input_width = (data[offset] % 8) + 1;
        offset++;
        int input_channels = (data[offset] % 8) + 1;
        offset++;
        
        int filter_depth = (data[offset] % 4) + 1;
        offset++;
        int filter_height = (data[offset] % 4) + 1;
        offset++;
        int filter_width = (data[offset] % 4) + 1;
        offset++;
        int output_channels = (data[offset] % 8) + 1;
        offset++;
        
        // Strides
        int stride_d = (data[offset] % 3) + 1;
        offset++;
        int stride_h = (data[offset] % 3) + 1;
        offset++;
        int stride_w = (data[offset] % 3) + 1;
        offset++;
        
        // Padding type
        bool use_same_padding = data[offset] % 2;
        offset++;
        
        // Data format
        bool use_ndhwc = data[offset] % 2;
        offset++;
        
        // Calculate output dimensions
        int out_depth, out_height, out_width;
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
        
        // Create input tensor (original input to conv3d)
        tensorflow::TensorShape input_shape;
        if (use_ndhwc) {
            input_shape = tensorflow::TensorShape({batch_size, input_depth, input_height, input_width, input_channels});
        } else {
            input_shape = tensorflow::TensorShape({batch_size, input_channels, input_depth, input_height, input_width});
        }
        
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, input_shape);
        auto input_flat = input_tensor.flat<float>();
        
        // Fill input tensor with fuzz data
        for (int i = 0; i < input_flat.size() && offset < size - 4; i++) {
            float val;
            memcpy(&val, data + offset, sizeof(float));
            input_flat(i) = val;
            offset += sizeof(float);
            if (offset >= size - 4) break;
        }
        
        // Create filter shape tensor
        tensorflow::Tensor filter_sizes(tensorflow::DT_INT32, tensorflow::TensorShape({5}));
        auto filter_sizes_flat = filter_sizes.flat<int32_t>();
        filter_sizes_flat(0) = filter_depth;
        filter_sizes_flat(1) = filter_height;
        filter_sizes_flat(2) = filter_width;
        filter_sizes_flat(3) = input_channels;
        filter_sizes_flat(4) = output_channels;
        
        // Create out_backprop tensor (gradient w.r.t. output)
        tensorflow::TensorShape out_backprop_shape;
        if (use_ndhwc) {
            out_backprop_shape = tensorflow::TensorShape({batch_size, out_depth, out_height, out_width, output_channels});
        } else {
            out_backprop_shape = tensorflow::TensorShape({batch_size, output_channels, out_depth, out_height, out_width});
        }
        
        tensorflow::Tensor out_backprop(tensorflow::DT_FLOAT, out_backprop_shape);
        auto out_backprop_flat = out_backprop.flat<float>();
        
        // Fill out_backprop with remaining fuzz data
        for (int i = 0; i < out_backprop_flat.size() && offset < size - 4; i++) {
            float val;
            if (offset + sizeof(float) <= size) {
                memcpy(&val, data + offset, sizeof(float));
                out_backprop_flat(i) = val;
                offset += sizeof(float);
            } else {
                out_backprop_flat(i) = 0.0f;
            }
        }
        
        // Create session and graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        
        node_def->set_name("conv3d_backprop_filter");
        node_def->set_op("Conv3DBackpropFilterV2");
        node_def->add_input("input");
        node_def->add_input("filter_sizes");
        node_def->add_input("out_backprop");
        
        // Set attributes
        tensorflow::AttrValue strides_attr;
        auto* strides_list = strides_attr.mutable_list();
        strides_list->add_i(1);
        strides_list->add_i(stride_d);
        strides_list->add_i(stride_h);
        strides_list->add_i(stride_w);
        strides_list->add_i(1);
        (*node_def->mutable_attr())["strides"] = strides_attr;
        
        tensorflow::AttrValue padding_attr;
        padding_attr.set_s(use_same_padding ? "SAME" : "VALID");
        (*node_def->mutable_attr())["padding"] = padding_attr;
        
        tensorflow::AttrValue data_format_attr;
        data_format_attr.set_s(use_ndhwc ? "NDHWC" : "NCDHW");
        (*node_def->mutable_attr())["data_format"] = data_format_attr;
        
        // Add input nodes
        tensorflow::NodeDef* input_node = graph_def.add_node();
        input_node->set_name("input");
        input_node->set_op("Placeholder");
        tensorflow::AttrValue input_dtype;
        input_dtype.set_type(tensorflow::DT_FLOAT);
        (*input_node->mutable_attr())["dtype"] = input_dtype;
        
        tensorflow::NodeDef* filter_sizes_node = graph_def.add_node();
        filter_sizes_node->set_name("filter_sizes");
        filter_sizes_node->set_op("Placeholder");
        tensorflow::AttrValue filter_sizes_dtype;
        filter_sizes_dtype.set_type(tensorflow::DT_INT32);
        (*filter_sizes_node->mutable_attr())["dtype"] = filter_sizes_dtype;
        
        tensorflow::NodeDef* out_backprop_node = graph_def.add_node();
        out_backprop_node->set_name("out_backprop");
        out_backprop_node->set_op("Placeholder");
        tensorflow::AttrValue out_backprop_dtype;
        out_backprop_dtype.set_type(tensorflow::DT_FLOAT);
        (*out_backprop_node->mutable_attr())["dtype"] = out_backprop_dtype;
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the operation
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"input", input_tensor},
            {"filter_sizes", filter_sizes},
            {"out_backprop", out_backprop}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"conv3d_backprop_filter"}, {}, &outputs);
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}