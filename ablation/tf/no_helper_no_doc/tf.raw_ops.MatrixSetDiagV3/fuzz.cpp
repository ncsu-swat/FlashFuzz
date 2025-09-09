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
        
        // Extract dimensions and parameters from fuzz input
        int batch_size = (data[offset] % 4) + 1;
        offset++;
        int matrix_rows = (data[offset] % 8) + 2;
        offset++;
        int matrix_cols = (data[offset] % 8) + 2;
        offset++;
        int num_diags = (data[offset] % 3) + 1;
        offset++;
        int k_lower = -(data[offset] % 3);
        offset++;
        int k_upper = k_lower + num_diags - 1;
        offset++;
        
        // Ensure we have enough data
        size_t required_size = batch_size * matrix_rows * matrix_cols * sizeof(float) +
                              batch_size * num_diags * std::min(matrix_rows, matrix_cols) * sizeof(float) +
                              2 * sizeof(int32_t);
        if (size < offset + required_size) return 0;
        
        // Create input tensor (matrix)
        tensorflow::TensorShape input_shape({batch_size, matrix_rows, matrix_cols});
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, input_shape);
        auto input_flat = input_tensor.flat<float>();
        
        // Fill input tensor with fuzz data
        for (int i = 0; i < input_flat.size() && offset + sizeof(float) <= size; i++) {
            float val;
            memcpy(&val, data + offset, sizeof(float));
            input_flat(i) = val;
            offset += sizeof(float);
        }
        
        // Create diagonal tensor
        int diag_size = batch_size * num_diags * std::min(matrix_rows, matrix_cols);
        tensorflow::TensorShape diag_shape({batch_size, num_diags, std::min(matrix_rows, matrix_cols)});
        tensorflow::Tensor diag_tensor(tensorflow::DT_FLOAT, diag_shape);
        auto diag_flat = diag_tensor.flat<float>();
        
        // Fill diagonal tensor with fuzz data
        for (int i = 0; i < diag_flat.size() && offset + sizeof(float) <= size; i++) {
            float val;
            memcpy(&val, data + offset, sizeof(float));
            diag_flat(i) = val;
            offset += sizeof(float);
        }
        
        // Create k tensor (diagonal offsets)
        tensorflow::Tensor k_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({num_diags}));
        auto k_flat = k_tensor.flat<int32_t>();
        for (int i = 0; i < num_diags; i++) {
            k_flat(i) = k_lower + i;
        }
        
        // Create session and graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        node_def->set_name("matrix_set_diag_v3");
        node_def->set_op("MatrixSetDiagV3");
        
        // Add input nodes
        tensorflow::NodeDef* input_node = graph_def.add_node();
        input_node->set_name("input");
        input_node->set_op("Placeholder");
        (*input_node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        
        tensorflow::NodeDef* diag_node = graph_def.add_node();
        diag_node->set_name("diagonal");
        diag_node->set_op("Placeholder");
        (*diag_node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
        
        tensorflow::NodeDef* k_node = graph_def.add_node();
        k_node->set_name("k");
        k_node->set_op("Placeholder");
        (*k_node->mutable_attr())["dtype"].set_type(tensorflow::DT_INT32);
        
        // Set up the MatrixSetDiagV3 node
        node_def->add_input("input");
        node_def->add_input("diagonal");
        node_def->add_input("k");
        (*node_def->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
        
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
            {"diagonal", diag_tensor},
            {"k", k_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"matrix_set_diag_v3"}, {}, &outputs);
        
        // Clean up
        session->Close();
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}