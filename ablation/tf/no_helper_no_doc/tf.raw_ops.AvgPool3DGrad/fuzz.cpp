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
        
        if (size < 32) return 0;
        
        // Extract dimensions for orig_input_shape (5D tensor: batch, depth, height, width, channels)
        int32_t batch = (data[offset] % 4) + 1; offset++;
        int32_t depth = (data[offset] % 8) + 1; offset++;
        int32_t height = (data[offset] % 16) + 1; offset++;
        int32_t width = (data[offset] % 16) + 1; offset++;
        int32_t channels = (data[offset] % 8) + 1; offset++;
        
        // Extract ksize (5D: batch, depth, height, width, channels)
        int32_t ksize_batch = 1;
        int32_t ksize_depth = (data[offset] % 4) + 1; offset++;
        int32_t ksize_height = (data[offset] % 4) + 1; offset++;
        int32_t ksize_width = (data[offset] % 4) + 1; offset++;
        int32_t ksize_channels = 1;
        
        // Extract strides (5D: batch, depth, height, width, channels)
        int32_t stride_batch = 1;
        int32_t stride_depth = (data[offset] % 3) + 1; offset++;
        int32_t stride_height = (data[offset] % 3) + 1; offset++;
        int32_t stride_width = (data[offset] % 3) + 1; offset++;
        int32_t stride_channels = 1;
        
        // Extract padding type
        std::string padding = (data[offset] % 2) ? "VALID" : "SAME"; offset++;
        
        // Extract data format
        std::string data_format = "NDHWC"; // Default format
        
        // Calculate output dimensions based on padding
        int32_t out_depth, out_height, out_width;
        if (padding == "VALID") {
            out_depth = (depth - ksize_depth) / stride_depth + 1;
            out_height = (height - ksize_height) / stride_height + 1;
            out_width = (width - ksize_width) / stride_width + 1;
        } else { // SAME
            out_depth = (depth + stride_depth - 1) / stride_depth;
            out_height = (height + stride_height - 1) / stride_height;
            out_width = (width + stride_width - 1) / stride_width;
        }
        
        if (out_depth <= 0 || out_height <= 0 || out_width <= 0) return 0;
        
        // Create orig_input_shape tensor
        tensorflow::Tensor orig_input_shape(tensorflow::DT_INT32, tensorflow::TensorShape({5}));
        auto orig_shape_flat = orig_input_shape.flat<int32_t>();
        orig_shape_flat(0) = batch;
        orig_shape_flat(1) = depth;
        orig_shape_flat(2) = height;
        orig_shape_flat(3) = width;
        orig_shape_flat(4) = channels;
        
        // Create grad tensor (output gradient)
        tensorflow::TensorShape grad_shape({batch, out_depth, out_height, out_width, channels});
        tensorflow::Tensor grad(tensorflow::DT_FLOAT, grad_shape);
        auto grad_flat = grad.flat<float>();
        
        // Fill grad tensor with fuzz data
        size_t grad_size = grad_flat.size();
        for (size_t i = 0; i < grad_size && offset < size; ++i) {
            grad_flat(i) = static_cast<float>(data[offset % size]) / 255.0f - 0.5f;
            offset++;
        }
        
        // Create session and graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        
        node_def->set_name("avg_pool_3d_grad");
        node_def->set_op("AvgPool3DGrad");
        node_def->add_input("orig_input_shape");
        node_def->add_input("grad");
        
        // Set attributes
        tensorflow::AttrValue attr_ksize;
        attr_ksize.mutable_list()->add_i(ksize_batch);
        attr_ksize.mutable_list()->add_i(ksize_depth);
        attr_ksize.mutable_list()->add_i(ksize_height);
        attr_ksize.mutable_list()->add_i(ksize_width);
        attr_ksize.mutable_list()->add_i(ksize_channels);
        (*node_def->mutable_attr())["ksize"] = attr_ksize;
        
        tensorflow::AttrValue attr_strides;
        attr_strides.mutable_list()->add_i(stride_batch);
        attr_strides.mutable_list()->add_i(stride_depth);
        attr_strides.mutable_list()->add_i(stride_height);
        attr_strides.mutable_list()->add_i(stride_width);
        attr_strides.mutable_list()->add_i(stride_channels);
        (*node_def->mutable_attr())["strides"] = attr_strides;
        
        tensorflow::AttrValue attr_padding;
        attr_padding.set_s(padding);
        (*node_def->mutable_attr())["padding"] = attr_padding;
        
        tensorflow::AttrValue attr_data_format;
        attr_data_format.set_s(data_format);
        (*node_def->mutable_attr())["data_format"] = attr_data_format;
        
        tensorflow::AttrValue attr_dtype;
        attr_dtype.set_type(tensorflow::DT_FLOAT);
        (*node_def->mutable_attr())["T"] = attr_dtype;
        
        // Add input nodes
        tensorflow::NodeDef* input_shape_node = graph_def.add_node();
        input_shape_node->set_name("orig_input_shape");
        input_shape_node->set_op("Placeholder");
        tensorflow::AttrValue input_shape_dtype;
        input_shape_dtype.set_type(tensorflow::DT_INT32);
        (*input_shape_node->mutable_attr())["dtype"] = input_shape_dtype;
        
        tensorflow::NodeDef* grad_node = graph_def.add_node();
        grad_node->set_name("grad");
        grad_node->set_op("Placeholder");
        tensorflow::AttrValue grad_dtype;
        grad_dtype.set_type(tensorflow::DT_FLOAT);
        (*grad_node->mutable_attr())["dtype"] = grad_dtype;
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the operation
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"orig_input_shape", orig_input_shape},
            {"grad", grad}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"avg_pool_3d_grad"}, {}, &outputs);
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}