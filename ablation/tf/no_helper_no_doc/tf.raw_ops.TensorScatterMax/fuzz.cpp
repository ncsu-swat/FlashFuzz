#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions and parameters from fuzz input
        uint32_t tensor_dim1 = *reinterpret_cast<const uint32_t*>(data + offset) % 10 + 1;
        offset += 4;
        uint32_t tensor_dim2 = *reinterpret_cast<const uint32_t*>(data + offset) % 10 + 1;
        offset += 4;
        uint32_t indices_rows = *reinterpret_cast<const uint32_t*>(data + offset) % 5 + 1;
        offset += 4;
        uint32_t updates_size = indices_rows * tensor_dim2;
        offset += 4;
        
        if (offset + tensor_dim1 * tensor_dim2 * sizeof(float) + 
            indices_rows * sizeof(int32_t) + 
            updates_size * sizeof(float) > size) {
            return 0;
        }
        
        // Create tensor shape
        tensorflow::TensorShape tensor_shape({tensor_dim1, tensor_dim2});
        tensorflow::TensorShape indices_shape({indices_rows, 1});
        tensorflow::TensorShape updates_shape({indices_rows, tensor_dim2});
        
        // Create input tensor
        tensorflow::Tensor tensor(tensorflow::DT_FLOAT, tensor_shape);
        auto tensor_flat = tensor.flat<float>();
        for (int i = 0; i < tensor_dim1 * tensor_dim2; ++i) {
            if (offset + sizeof(float) <= size) {
                tensor_flat(i) = *reinterpret_cast<const float*>(data + offset);
                offset += sizeof(float);
            } else {
                tensor_flat(i) = 0.0f;
            }
        }
        
        // Create indices tensor
        tensorflow::Tensor indices(tensorflow::DT_INT32, indices_shape);
        auto indices_flat = indices.flat<int32_t>();
        for (int i = 0; i < indices_rows; ++i) {
            if (offset + sizeof(int32_t) <= size) {
                indices_flat(i) = (*reinterpret_cast<const int32_t*>(data + offset)) % tensor_dim1;
                if (indices_flat(i) < 0) indices_flat(i) = 0;
                offset += sizeof(int32_t);
            } else {
                indices_flat(i) = 0;
            }
        }
        
        // Create updates tensor
        tensorflow::Tensor updates(tensorflow::DT_FLOAT, updates_shape);
        auto updates_flat = updates.flat<float>();
        for (int i = 0; i < updates_size; ++i) {
            if (offset + sizeof(float) <= size) {
                updates_flat(i) = *reinterpret_cast<const float*>(data + offset);
                offset += sizeof(float);
            } else {
                updates_flat(i) = 0.0f;
            }
        }
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Build graph
        tensorflow::GraphDefBuilder builder(tensorflow::GraphDefBuilder::kFailImmediately);
        
        auto tensor_node = tensorflow::ops::Const(tensor, builder.opts().WithName("tensor"));
        auto indices_node = tensorflow::ops::Const(indices, builder.opts().WithName("indices"));
        auto updates_node = tensorflow::ops::Const(updates, builder.opts().WithName("updates"));
        
        auto scatter_max = tensorflow::ops::TensorScatterMax(
            tensor_node, indices_node, updates_node, builder.opts().WithName("scatter_max"));
        
        tensorflow::GraphDef graph_def;
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
        status = session->Run({}, {"scatter_max:0"}, {}, &outputs);
        if (!status.ok()) {
            return 0;
        }
        
        // Verify output shape matches input tensor shape
        if (outputs.size() > 0 && outputs[0].shape() == tensor_shape) {
            // Operation completed successfully
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}