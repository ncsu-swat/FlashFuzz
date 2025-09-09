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
        
        if (size < 16) return 0;
        
        // Extract parameters from fuzzer input
        int32_t num_indices = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        int32_t num_values = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        int32_t dense_shape_size = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        int32_t size_val = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        // Clamp values to reasonable ranges
        num_indices = std::max(1, std::min(num_indices, 100));
        num_values = std::max(1, std::min(num_values, 100));
        dense_shape_size = std::max(1, std::min(dense_shape_size, 10));
        size_val = std::max(1, std::min(size_val, 1000));
        
        // Check if we have enough data
        size_t required_size = offset + 
                              (num_indices * 2 * sizeof(int64_t)) + // indices (2D)
                              (num_values * sizeof(int32_t)) +      // values
                              (dense_shape_size * sizeof(int64_t)) + // dense_shape
                              sizeof(int32_t);                      // size
        
        if (size < required_size) return 0;
        
        // Create TensorFlow session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create graph
        tensorflow::GraphDef graph_def;
        
        // Create indices tensor (2D: [num_indices, 2])
        auto indices_node = graph_def.add_node();
        indices_node->set_name("indices");
        indices_node->set_op("Const");
        (*indices_node->mutable_attr())["dtype"].set_type(tensorflow::DT_INT64);
        
        tensorflow::TensorProto indices_proto;
        indices_proto.set_dtype(tensorflow::DT_INT64);
        indices_proto.mutable_tensor_shape()->add_dim()->set_size(num_indices);
        indices_proto.mutable_tensor_shape()->add_dim()->set_size(2);
        
        for (int i = 0; i < num_indices * 2; ++i) {
            if (offset + sizeof(int64_t) <= size) {
                int64_t val = *reinterpret_cast<const int64_t*>(data + offset);
                val = std::max(0LL, std::min(val, 99LL)); // Clamp to valid range
                indices_proto.add_int64_val(val);
                offset += sizeof(int64_t);
            } else {
                indices_proto.add_int64_val(0);
            }
        }
        (*indices_node->mutable_attr())["value"].mutable_tensor()->CopyFrom(indices_proto);
        
        // Create values tensor
        auto values_node = graph_def.add_node();
        values_node->set_name("values");
        values_node->set_op("Const");
        (*values_node->mutable_attr())["dtype"].set_type(tensorflow::DT_INT32);
        
        tensorflow::TensorProto values_proto;
        values_proto.set_dtype(tensorflow::DT_INT32);
        values_proto.mutable_tensor_shape()->add_dim()->set_size(num_values);
        
        for (int i = 0; i < num_values; ++i) {
            if (offset + sizeof(int32_t) <= size) {
                int32_t val = *reinterpret_cast<const int32_t*>(data + offset);
                val = std::max(0, std::min(val, 999)); // Clamp to valid range
                values_proto.add_int_val(val);
                offset += sizeof(int32_t);
            } else {
                values_proto.add_int_val(0);
            }
        }
        (*values_node->mutable_attr())["value"].mutable_tensor()->CopyFrom(values_proto);
        
        // Create dense_shape tensor
        auto dense_shape_node = graph_def.add_node();
        dense_shape_node->set_name("dense_shape");
        dense_shape_node->set_op("Const");
        (*dense_shape_node->mutable_attr())["dtype"].set_type(tensorflow::DT_INT64);
        
        tensorflow::TensorProto dense_shape_proto;
        dense_shape_proto.set_dtype(tensorflow::DT_INT64);
        dense_shape_proto.mutable_tensor_shape()->add_dim()->set_size(dense_shape_size);
        
        for (int i = 0; i < dense_shape_size; ++i) {
            if (offset + sizeof(int64_t) <= size) {
                int64_t val = *reinterpret_cast<const int64_t*>(data + offset);
                val = std::max(1LL, std::min(val, 100LL)); // Clamp to valid range
                dense_shape_proto.add_int64_val(val);
                offset += sizeof(int64_t);
            } else {
                dense_shape_proto.add_int64_val(10);
            }
        }
        (*dense_shape_node->mutable_attr())["dense_shape"].mutable_tensor()->CopyFrom(dense_shape_proto);
        
        // Create size tensor
        auto size_node = graph_def.add_node();
        size_node->set_name("size");
        size_node->set_op("Const");
        (*size_node->mutable_attr())["dtype"].set_type(tensorflow::DT_INT32);
        
        tensorflow::TensorProto size_proto;
        size_proto.set_dtype(tensorflow::DT_INT32);
        size_proto.mutable_tensor_shape(); // scalar
        size_proto.add_int_val(size_val);
        (*size_node->mutable_attr())["value"].mutable_tensor()->CopyFrom(size_proto);
        
        // Create SparseBincount node
        auto bincount_node = graph_def.add_node();
        bincount_node->set_name("sparse_bincount");
        bincount_node->set_op("SparseBincount");
        bincount_node->add_input("indices");
        bincount_node->add_input("values");
        bincount_node->add_input("dense_shape");
        bincount_node->add_input("size");
        (*bincount_node->mutable_attr())["T"].set_type(tensorflow::DT_INT32);
        
        // Create the session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({}, {"sparse_bincount:0"}, {}, &outputs);
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}