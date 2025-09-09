#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/framework/node_def_util.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions for input matrix
        int32_t batch_size = (data[offset] % 4) + 1;
        offset++;
        int32_t rows = (data[offset] % 8) + 2;
        offset++;
        int32_t cols = (data[offset] % 8) + 2;
        offset++;
        
        // Extract k parameter (diagonal offset)
        int32_t k = static_cast<int32_t>(data[offset] % 5) - 2; // Range [-2, 2]
        offset++;
        
        // Calculate diagonal length
        int32_t diag_len;
        if (k >= 0) {
            diag_len = std::min(rows, cols - k);
        } else {
            diag_len = std::min(rows + k, cols);
        }
        diag_len = std::max(diag_len, 0);
        
        if (diag_len == 0) return 0;
        
        // Create input tensor (matrix)
        tensorflow::TensorShape input_shape({batch_size, rows, cols});
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, input_shape);
        auto input_flat = input_tensor.flat<float>();
        
        // Fill input tensor with fuzz data
        size_t input_elements = batch_size * rows * cols;
        for (size_t i = 0; i < input_elements && offset + 4 <= size; i++) {
            float val;
            memcpy(&val, data + offset, sizeof(float));
            input_flat(i) = val;
            offset += 4;
        }
        
        // Create diagonal tensor
        tensorflow::TensorShape diag_shape({batch_size, diag_len});
        tensorflow::Tensor diag_tensor(tensorflow::DT_FLOAT, diag_shape);
        auto diag_flat = diag_tensor.flat<float>();
        
        // Fill diagonal tensor with fuzz data
        size_t diag_elements = batch_size * diag_len;
        for (size_t i = 0; i < diag_elements && offset + 4 <= size; i++) {
            float val;
            memcpy(&val, data + offset, sizeof(float));
            diag_flat(i) = val;
            offset += 4;
        }
        
        // Create k tensor
        tensorflow::Tensor k_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
        k_tensor.scalar<int32_t>()() = k;
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Build graph
        tensorflow::GraphDef graph_def;
        tensorflow::GraphDefBuilder builder(tensorflow::GraphDefBuilder::kFailImmediately);
        
        auto input_node = tensorflow::ops::Placeholder(builder.opts()
            .WithName("input")
            .WithAttr("dtype", tensorflow::DT_FLOAT)
            .WithAttr("shape", input_shape));
            
        auto diag_node = tensorflow::ops::Placeholder(builder.opts()
            .WithName("diagonal")
            .WithAttr("dtype", tensorflow::DT_FLOAT)
            .WithAttr("shape", diag_shape));
            
        auto k_node = tensorflow::ops::Placeholder(builder.opts()
            .WithName("k")
            .WithAttr("dtype", tensorflow::DT_INT32)
            .WithAttr("shape", tensorflow::TensorShape({})));
        
        // Create MatrixSetDiag operation
        tensorflow::NodeDef matrix_set_diag_node;
        matrix_set_diag_node.set_name("matrix_set_diag");
        matrix_set_diag_node.set_op("MatrixSetDiagV3");
        matrix_set_diag_node.add_input("input");
        matrix_set_diag_node.add_input("diagonal");
        matrix_set_diag_node.add_input("k");
        (*matrix_set_diag_node.mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
        (*matrix_set_diag_node.mutable_attr())["Tindex"].set_type(tensorflow::DT_INT32);
        
        *graph_def.add_node() = input_node;
        *graph_def.add_node() = diag_node;
        *graph_def.add_node() = k_node;
        *graph_def.add_node() = matrix_set_diag_node;
        
        // Create session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"input", input_tensor},
            {"diagonal", diag_tensor},
            {"k", k_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {"matrix_set_diag"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Verify output shape matches input shape
            if (outputs[0].shape() == input_shape) {
                // Basic validation passed
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