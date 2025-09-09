#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/kernels/ops_util.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 32) return 0;
        
        // Extract dimensions and parameters from fuzz input
        int64_t num_dims = (data[offset] % 3) + 1; // 1-3 dimensions
        offset++;
        
        int64_t nnz_a = (data[offset] % 10) + 1; // 1-10 non-zero elements for sparse tensor A
        offset++;
        
        int64_t nnz_b = (data[offset] % 10) + 1; // 1-10 non-zero elements for sparse tensor B
        offset++;
        
        if (offset + num_dims * 8 + nnz_a * (num_dims * 8 + 4) + nnz_b * (num_dims * 8 + 4) > size) {
            return 0;
        }
        
        // Create shape tensor
        std::vector<int64_t> shape_data(num_dims);
        for (int i = 0; i < num_dims; i++) {
            shape_data[i] = ((data[offset] % 10) + 1) * 2; // Ensure positive shape
            offset++;
        }
        
        tensorflow::Tensor shape_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({num_dims}));
        auto shape_flat = shape_tensor.flat<int64_t>();
        for (int i = 0; i < num_dims; i++) {
            shape_flat(i) = shape_data[i];
        }
        
        // Create sparse tensor A indices
        tensorflow::Tensor a_indices(tensorflow::DT_INT64, tensorflow::TensorShape({nnz_a, num_dims}));
        auto a_indices_matrix = a_indices.matrix<int64_t>();
        for (int i = 0; i < nnz_a; i++) {
            for (int j = 0; j < num_dims; j++) {
                if (offset >= size) return 0;
                a_indices_matrix(i, j) = data[offset] % shape_data[j];
                offset++;
            }
        }
        
        // Create sparse tensor A values
        tensorflow::Tensor a_values(tensorflow::DT_FLOAT, tensorflow::TensorShape({nnz_a}));
        auto a_values_flat = a_values.flat<float>();
        for (int i = 0; i < nnz_a; i++) {
            if (offset + 3 >= size) return 0;
            float val = *reinterpret_cast<const float*>(data + offset);
            if (std::isnan(val) || std::isinf(val)) {
                val = 1.0f;
            }
            a_values_flat(i) = val;
            offset += 4;
        }
        
        // Create sparse tensor B indices
        tensorflow::Tensor b_indices(tensorflow::DT_INT64, tensorflow::TensorShape({nnz_b, num_dims}));
        auto b_indices_matrix = b_indices.matrix<int64_t>();
        for (int i = 0; i < nnz_b; i++) {
            for (int j = 0; j < num_dims; j++) {
                if (offset >= size) return 0;
                b_indices_matrix(i, j) = data[offset] % shape_data[j];
                offset++;
            }
        }
        
        // Create sparse tensor B values
        tensorflow::Tensor b_values(tensorflow::DT_FLOAT, tensorflow::TensorShape({nnz_b}));
        auto b_values_flat = b_values.flat<float>();
        for (int i = 0; i < nnz_b; i++) {
            if (offset + 3 >= size) return 0;
            float val = *reinterpret_cast<const float*>(data + offset);
            if (std::isnan(val) || std::isinf(val)) {
                val = 1.0f;
            }
            b_values_flat(i) = val;
            offset += 4;
        }
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Build graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        
        node_def->set_name("sparse_sparse_maximum");
        node_def->set_op("SparseSparseMaximum");
        
        // Add input names
        node_def->add_input("a_indices:0");
        node_def->add_input("a_values:0");
        node_def->add_input("a_shape:0");
        node_def->add_input("b_indices:0");
        node_def->add_input("b_values:0");
        node_def->add_input("b_shape:0");
        
        // Set attributes
        (*node_def->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
        
        // Add placeholder nodes for inputs
        auto add_placeholder = [&](const std::string& name, tensorflow::DataType dtype, const tensorflow::TensorShape& shape) {
            tensorflow::NodeDef* placeholder = graph_def.add_node();
            placeholder->set_name(name);
            placeholder->set_op("Placeholder");
            (*placeholder->mutable_attr())["dtype"].set_type(dtype);
            (*placeholder->mutable_attr())["shape"].mutable_shape()->CopyFrom(shape.AsProto());
        };
        
        add_placeholder("a_indices", tensorflow::DT_INT64, a_indices.shape());
        add_placeholder("a_values", tensorflow::DT_FLOAT, a_values.shape());
        add_placeholder("a_shape", tensorflow::DT_INT64, shape_tensor.shape());
        add_placeholder("b_indices", tensorflow::DT_INT64, b_indices.shape());
        add_placeholder("b_values", tensorflow::DT_FLOAT, b_values.shape());
        add_placeholder("b_shape", tensorflow::DT_INT64, shape_tensor.shape());
        
        // Create session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"a_indices:0", a_indices},
            {"a_values:0", a_values},
            {"a_shape:0", shape_tensor},
            {"b_indices:0", b_indices},
            {"b_values:0", b_values},
            {"b_shape:0", shape_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {
            "sparse_sparse_maximum:0",
            "sparse_sparse_maximum:1",
            "sparse_sparse_maximum:2"
        };
        
        status = session->Run(inputs, output_names, {}, &outputs);
        
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}