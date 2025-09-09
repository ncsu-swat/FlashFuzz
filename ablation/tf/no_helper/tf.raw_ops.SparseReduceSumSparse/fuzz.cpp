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
        
        // Extract dimensions from fuzz data
        uint32_t num_indices = (data[offset] % 10) + 1;
        offset++;
        uint32_t rank = (data[offset] % 5) + 1;
        offset++;
        uint32_t num_reduction_axes = (data[offset] % rank) + 1;
        offset++;
        bool keep_dims = data[offset] % 2;
        offset++;
        
        if (offset + num_indices * rank * 8 + num_indices * 4 + rank * 8 + num_reduction_axes * 4 > size) {
            return 0;
        }
        
        // Create input_indices tensor (N x R matrix)
        tensorflow::Tensor input_indices(tensorflow::DT_INT64, 
                                       tensorflow::TensorShape({static_cast<int64_t>(num_indices), static_cast<int64_t>(rank)}));
        auto indices_matrix = input_indices.matrix<int64_t>();
        for (int i = 0; i < num_indices; i++) {
            for (int j = 0; j < rank; j++) {
                int64_t val;
                memcpy(&val, data + offset, sizeof(int64_t));
                indices_matrix(i, j) = std::abs(val) % 10; // Keep indices small and positive
                offset += sizeof(int64_t);
            }
        }
        
        // Create input_values tensor (N values)
        tensorflow::Tensor input_values(tensorflow::DT_FLOAT, 
                                      tensorflow::TensorShape({static_cast<int64_t>(num_indices)}));
        auto values_vec = input_values.vec<float>();
        for (int i = 0; i < num_indices; i++) {
            float val;
            memcpy(&val, data + offset, sizeof(float));
            values_vec(i) = val;
            offset += sizeof(float);
        }
        
        // Create input_shape tensor (R dimensions)
        tensorflow::Tensor input_shape(tensorflow::DT_INT64, 
                                     tensorflow::TensorShape({static_cast<int64_t>(rank)}));
        auto shape_vec = input_shape.vec<int64_t>();
        for (int i = 0; i < rank; i++) {
            int64_t val;
            memcpy(&val, data + offset, sizeof(int64_t));
            shape_vec(i) = (std::abs(val) % 10) + 1; // Keep shape dimensions positive and small
            offset += sizeof(int64_t);
        }
        
        // Create reduction_axes tensor
        tensorflow::Tensor reduction_axes(tensorflow::DT_INT32, 
                                        tensorflow::TensorShape({static_cast<int64_t>(num_reduction_axes)}));
        auto axes_vec = reduction_axes.vec<int32_t>();
        for (int i = 0; i < num_reduction_axes; i++) {
            int32_t val;
            memcpy(&val, data + offset, sizeof(int32_t));
            axes_vec(i) = val % rank; // Keep axes within valid range
            offset += sizeof(int32_t);
        }
        
        // Create session and graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        
        node_def->set_name("sparse_reduce_sum_sparse");
        node_def->set_op("SparseReduceSumSparse");
        
        // Add inputs
        node_def->add_input("input_indices:0");
        node_def->add_input("input_values:0");
        node_def->add_input("input_shape:0");
        node_def->add_input("reduction_axes:0");
        
        // Set keep_dims attribute
        tensorflow::AttrValue keep_dims_attr;
        keep_dims_attr.set_b(keep_dims);
        (*node_def->mutable_attr())["keep_dims"] = keep_dims_attr;
        
        // Add placeholder nodes for inputs
        auto add_placeholder = [&](const std::string& name, tensorflow::DataType dtype, const tensorflow::TensorShape& shape) {
            tensorflow::NodeDef* placeholder = graph_def.add_node();
            placeholder->set_name(name);
            placeholder->set_op("Placeholder");
            tensorflow::AttrValue dtype_attr;
            dtype_attr.set_type(dtype);
            (*placeholder->mutable_attr())["dtype"] = dtype_attr;
            tensorflow::AttrValue shape_attr;
            shape.AsProto(shape_attr.mutable_shape());
            (*placeholder->mutable_attr())["shape"] = shape_attr;
        };
        
        add_placeholder("input_indices", tensorflow::DT_INT64, input_indices.shape());
        add_placeholder("input_values", tensorflow::DT_FLOAT, input_values.shape());
        add_placeholder("input_shape", tensorflow::DT_INT64, input_shape.shape());
        add_placeholder("reduction_axes", tensorflow::DT_INT32, reduction_axes.shape());
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Prepare inputs
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"input_indices:0", input_indices},
            {"input_values:0", input_values},
            {"input_shape:0", input_shape},
            {"reduction_axes:0", reduction_axes}
        };
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {
            "sparse_reduce_sum_sparse:0",
            "sparse_reduce_sum_sparse:1", 
            "sparse_reduce_sum_sparse:2"
        };
        
        status = session->Run(inputs, output_names, {}, &outputs);
        
        // Clean up
        session->Close();
        
        if (status.ok() && outputs.size() == 3) {
            // Verify output tensors have expected types
            if (outputs[0].dtype() == tensorflow::DT_INT64 &&
                outputs[1].dtype() == tensorflow::DT_FLOAT &&
                outputs[2].dtype() == tensorflow::DT_INT64) {
                // Success - outputs have correct types
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}