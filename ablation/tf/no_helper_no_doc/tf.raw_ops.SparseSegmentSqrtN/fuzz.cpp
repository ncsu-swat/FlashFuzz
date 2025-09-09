#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/kernels/sparse_segment_reduction_ops.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/str_util.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/framework/node_def_util.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/kernels/ops_util.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions from fuzz data
        int32_t data_rows = (data[offset] % 10) + 1;
        offset++;
        int32_t data_cols = (data[offset] % 10) + 1;
        offset++;
        int32_t num_indices = (data[offset] % data_rows) + 1;
        offset++;
        int32_t num_segments = (data[offset] % num_indices) + 1;
        offset++;
        
        if (offset + data_rows * data_cols * sizeof(float) + 
            num_indices * sizeof(int32_t) + 
            num_indices * sizeof(int32_t) > size) {
            return 0;
        }
        
        // Create input data tensor
        tensorflow::Tensor data_tensor(tensorflow::DT_FLOAT, 
                                     tensorflow::TensorShape({data_rows, data_cols}));
        auto data_flat = data_tensor.flat<float>();
        for (int i = 0; i < data_rows * data_cols; i++) {
            if (offset + sizeof(float) > size) return 0;
            float val;
            memcpy(&val, data + offset, sizeof(float));
            data_flat(i) = val;
            offset += sizeof(float);
        }
        
        // Create indices tensor
        tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, 
                                        tensorflow::TensorShape({num_indices}));
        auto indices_flat = indices_tensor.flat<int32_t>();
        for (int i = 0; i < num_indices; i++) {
            if (offset + sizeof(int32_t) > size) return 0;
            int32_t val;
            memcpy(&val, data + offset, sizeof(int32_t));
            indices_flat(i) = abs(val) % data_rows;
            offset += sizeof(int32_t);
        }
        
        // Create segment_ids tensor
        tensorflow::Tensor segment_ids_tensor(tensorflow::DT_INT32, 
                                             tensorflow::TensorShape({num_indices}));
        auto segment_ids_flat = segment_ids_tensor.flat<int32_t>();
        int32_t current_segment = 0;
        for (int i = 0; i < num_indices; i++) {
            segment_ids_flat(i) = current_segment;
            if (i > 0 && (data[offset % size] % 3) == 0 && current_segment < num_segments - 1) {
                current_segment++;
            }
            offset++;
        }
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Build graph
        tensorflow::GraphDef graph_def;
        tensorflow::GraphDefBuilder builder(tensorflow::GraphDefBuilder::kFailImmediately);
        
        auto data_node = tensorflow::ops::Const(data_tensor, builder.opts().WithName("data"));
        auto indices_node = tensorflow::ops::Const(indices_tensor, builder.opts().WithName("indices"));
        auto segment_ids_node = tensorflow::ops::Const(segment_ids_tensor, builder.opts().WithName("segment_ids"));
        
        tensorflow::NodeDef sparse_segment_sqrt_n_node;
        sparse_segment_sqrt_n_node.set_name("sparse_segment_sqrt_n");
        sparse_segment_sqrt_n_node.set_op("SparseSegmentSqrtN");
        sparse_segment_sqrt_n_node.add_input("data");
        sparse_segment_sqrt_n_node.add_input("indices");
        sparse_segment_sqrt_n_node.add_input("segment_ids");
        tensorflow::AddNodeAttr("T", tensorflow::DT_FLOAT, &sparse_segment_sqrt_n_node);
        tensorflow::AddNodeAttr("Tidx", tensorflow::DT_INT32, &sparse_segment_sqrt_n_node);
        
        *graph_def.add_node() = sparse_segment_sqrt_n_node;
        
        // Add input nodes to graph
        *graph_def.add_node() = data_node.node();
        *graph_def.add_node() = indices_node.node();
        *graph_def.add_node() = segment_ids_node.node();
        
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({}, {"sparse_segment_sqrt_n:0"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Verify output shape
            auto output_shape = outputs[0].shape();
            if (output_shape.dims() == 2 && 
                output_shape.dim_size(0) > 0 && 
                output_shape.dim_size(1) == data_cols) {
                // Operation succeeded
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