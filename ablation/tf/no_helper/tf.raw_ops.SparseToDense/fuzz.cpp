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
        
        if (size < 20) return 0;
        
        // Extract basic parameters
        uint8_t indices_rank = data[offset++] % 3; // 0, 1, or 2
        uint8_t output_dim = (data[offset++] % 4) + 1; // 1-4 dimensions
        uint8_t num_sparse = (data[offset++] % 8) + 1; // 1-8 sparse values
        bool validate_indices = data[offset++] % 2;
        bool use_int64 = data[offset++] % 2;
        bool scalar_values = data[offset++] % 2;
        
        if (offset + 14 > size) return 0;
        
        // Create TensorFlow session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        tensorflow::GraphDef graph;
        
        // Create output shape tensor
        tensorflow::TensorShape output_shape_tensor_shape({output_dim});
        tensorflow::Tensor output_shape_tensor(use_int64 ? tensorflow::DT_INT64 : tensorflow::DT_INT32, 
                                              output_shape_tensor_shape);
        
        if (use_int64) {
            auto output_shape_flat = output_shape_tensor.flat<int64_t>();
            for (int i = 0; i < output_dim; i++) {
                output_shape_flat(i) = (data[offset++] % 10) + 1; // 1-10 size per dim
            }
        } else {
            auto output_shape_flat = output_shape_tensor.flat<int32_t>();
            for (int i = 0; i < output_dim; i++) {
                output_shape_flat(i) = (data[offset++] % 10) + 1; // 1-10 size per dim
            }
        }
        
        // Create sparse indices tensor
        tensorflow::TensorShape indices_shape;
        if (indices_rank == 0) {
            indices_shape = tensorflow::TensorShape({});
        } else if (indices_rank == 1) {
            indices_shape = tensorflow::TensorShape({num_sparse});
        } else {
            indices_shape = tensorflow::TensorShape({num_sparse, output_dim});
        }
        
        tensorflow::Tensor sparse_indices_tensor(use_int64 ? tensorflow::DT_INT64 : tensorflow::DT_INT32, 
                                                 indices_shape);
        
        if (use_int64) {
            auto indices_flat = sparse_indices_tensor.flat<int64_t>();
            for (int i = 0; i < indices_flat.size(); i++) {
                if (offset >= size) break;
                indices_flat(i) = data[offset++] % 5; // Keep indices small
            }
        } else {
            auto indices_flat = sparse_indices_tensor.flat<int32_t>();
            for (int i = 0; i < indices_flat.size(); i++) {
                if (offset >= size) break;
                indices_flat(i) = data[offset++] % 5; // Keep indices small
            }
        }
        
        // Create sparse values tensor
        tensorflow::TensorShape values_shape;
        if (scalar_values) {
            values_shape = tensorflow::TensorShape({});
        } else {
            values_shape = tensorflow::TensorShape({num_sparse});
        }
        
        tensorflow::Tensor sparse_values_tensor(tensorflow::DT_FLOAT, values_shape);
        auto values_flat = sparse_values_tensor.flat<float>();
        for (int i = 0; i < values_flat.size(); i++) {
            if (offset >= size) break;
            values_flat(i) = static_cast<float>(data[offset++]) / 255.0f;
        }
        
        // Create default value tensor (scalar)
        tensorflow::Tensor default_value_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        if (offset < size) {
            default_value_tensor.scalar<float>()() = static_cast<float>(data[offset++]) / 255.0f;
        } else {
            default_value_tensor.scalar<float>()() = 0.0f;
        }
        
        // Build the graph
        auto root = tensorflow::Scope::NewRootScope();
        
        auto sparse_indices_const = tensorflow::ops::Const(root, sparse_indices_tensor);
        auto output_shape_const = tensorflow::ops::Const(root, output_shape_tensor);
        auto sparse_values_const = tensorflow::ops::Const(root, sparse_values_tensor);
        auto default_value_const = tensorflow::ops::Const(root, default_value_tensor);
        
        // Create SparseToDense operation
        tensorflow::Node* sparse_to_dense_node;
        tensorflow::NodeBuilder builder("sparse_to_dense", "SparseToDense");
        builder.Input(sparse_indices_const.node())
               .Input(output_shape_const.node())
               .Input(sparse_values_const.node())
               .Input(default_value_const.node())
               .Attr("validate_indices", validate_indices);
        
        tensorflow::Status status = builder.Finalize(root.graph(), &sparse_to_dense_node);
        if (!status.ok()) {
            return 0;
        }
        
        // Convert to GraphDef and run
        status = root.ToGraphDef(&graph);
        if (!status.ok()) {
            return 0;
        }
        
        status = session->Create(graph);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({}, {"sparse_to_dense"}, {}, &outputs);
        
        session->Close();
        
        // If execution succeeds, we've successfully tested the operation
        if (status.ok() && !outputs.empty()) {
            // Optionally verify output properties
            const auto& output = outputs[0];
            if (output.dims() == output_dim) {
                // Basic sanity check passed
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}