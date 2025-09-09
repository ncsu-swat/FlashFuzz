#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/framework/kernel_def_builder.h>
#include <tensorflow/core/kernels/ops_testutil.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/status_test_util.h>
#include <tensorflow/core/platform/test.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions and data type
        uint8_t dtype_idx = data[offset++] % 11; // 11 supported types
        uint8_t x_dims = (data[offset++] % 4) + 1; // 1-4 dimensions
        uint8_t y_dims = (data[offset++] % 4) + 1; // 1-4 dimensions
        
        tensorflow::DataType dtype;
        switch (dtype_idx) {
            case 0: dtype = tensorflow::DT_FLOAT; break;
            case 1: dtype = tensorflow::DT_DOUBLE; break;
            case 2: dtype = tensorflow::DT_INT32; break;
            case 3: dtype = tensorflow::DT_INT64; break;
            case 4: dtype = tensorflow::DT_INT8; break;
            case 5: dtype = tensorflow::DT_INT16; break;
            case 6: dtype = tensorflow::DT_UINT8; break;
            case 7: dtype = tensorflow::DT_UINT16; break;
            case 8: dtype = tensorflow::DT_UINT32; break;
            case 9: dtype = tensorflow::DT_UINT64; break;
            case 10: dtype = tensorflow::DT_BFLOAT16; break;
            default: dtype = tensorflow::DT_FLOAT; break;
        }
        
        // Create tensor shapes
        tensorflow::TensorShape x_shape, y_shape;
        for (int i = 0; i < x_dims && offset < size; i++) {
            int dim_size = (data[offset++] % 8) + 1; // 1-8 elements per dim
            x_shape.AddDim(dim_size);
        }
        for (int i = 0; i < y_dims && offset < size; i++) {
            int dim_size = (data[offset++] % 8) + 1; // 1-8 elements per dim
            y_shape.AddDim(dim_size);
        }
        
        if (x_shape.num_elements() == 0 || y_shape.num_elements() == 0) return 0;
        
        // Create tensors
        tensorflow::Tensor x_tensor(dtype, x_shape);
        tensorflow::Tensor y_tensor(dtype, y_shape);
        
        // Fill tensors with fuzz data
        size_t x_bytes = x_tensor.TotalBytes();
        size_t y_bytes = y_tensor.TotalBytes();
        
        if (offset + x_bytes + y_bytes > size) return 0;
        
        std::memcpy(x_tensor.tensor_data().data(), data + offset, std::min(x_bytes, size - offset));
        offset += x_bytes;
        
        if (offset < size) {
            std::memcpy(y_tensor.tensor_data().data(), data + offset, std::min(y_bytes, size - offset));
        }
        
        // Ensure y tensor doesn't contain zeros to avoid division by zero
        if (dtype == tensorflow::DT_FLOAT) {
            auto y_flat = y_tensor.flat<float>();
            for (int i = 0; i < y_flat.size(); i++) {
                if (y_flat(i) == 0.0f) y_flat(i) = 1.0f;
            }
        } else if (dtype == tensorflow::DT_DOUBLE) {
            auto y_flat = y_tensor.flat<double>();
            for (int i = 0; i < y_flat.size(); i++) {
                if (y_flat(i) == 0.0) y_flat(i) = 1.0;
            }
        } else if (dtype == tensorflow::DT_INT32) {
            auto y_flat = y_tensor.flat<int32_t>();
            for (int i = 0; i < y_flat.size(); i++) {
                if (y_flat(i) == 0) y_flat(i) = 1;
            }
        } else if (dtype == tensorflow::DT_INT64) {
            auto y_flat = y_tensor.flat<int64_t>();
            for (int i = 0; i < y_flat.size(); i++) {
                if (y_flat(i) == 0) y_flat(i) = 1;
            }
        }
        
        // Create session and graph
        tensorflow::GraphDefBuilder builder(tensorflow::GraphDefBuilder::kFailImmediately);
        
        auto x_node = tensorflow::ops::Const(x_tensor, builder.opts().WithName("x"));
        auto y_node = tensorflow::ops::Const(y_tensor, builder.opts().WithName("y"));
        
        tensorflow::NodeDefBuilder node_builder("truncate_div", "TruncateDiv");
        node_builder.Input(x_node.name(), 0, dtype);
        node_builder.Input(y_node.name(), 0, dtype);
        node_builder.Attr("T", dtype);
        
        tensorflow::NodeDef node_def;
        auto status = node_builder.Finalize(&node_def);
        if (!status.ok()) return 0;
        
        builder.opts().FinalizeGraph();
        tensorflow::GraphDef graph_def;
        status = builder.ToGraphDef(&graph_def);
        if (!status.ok()) return 0;
        
        // Add the TruncateDiv node to the graph
        *graph_def.add_node() = node_def;
        
        // Create session and run
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        status = session->Create(graph_def);
        if (!status.ok()) return 0;
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({}, {"truncate_div:0"}, {}, &outputs);
        if (!status.ok()) return 0;
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}