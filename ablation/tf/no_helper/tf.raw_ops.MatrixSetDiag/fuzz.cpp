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
        
        // Extract dimensions from fuzz data
        uint32_t batch_dim = (data[offset] % 3) + 1; // 1-3 batch dimensions
        offset += 1;
        
        uint32_t m = (data[offset] % 10) + 1; // matrix height 1-10
        offset += 1;
        
        uint32_t n = (data[offset] % 10) + 1; // matrix width 1-10
        offset += 1;
        
        uint32_t data_type_idx = data[offset] % 3; // 0=float, 1=double, 2=int32
        offset += 1;
        
        tensorflow::DataType dtype;
        size_t element_size;
        switch (data_type_idx) {
            case 0:
                dtype = tensorflow::DT_FLOAT;
                element_size = sizeof(float);
                break;
            case 1:
                dtype = tensorflow::DT_DOUBLE;
                element_size = sizeof(double);
                break;
            default:
                dtype = tensorflow::DT_INT32;
                element_size = sizeof(int32_t);
                break;
        }
        
        // Create input tensor shape [batch_dim, m, n]
        tensorflow::TensorShape input_shape;
        input_shape.AddDim(batch_dim);
        input_shape.AddDim(m);
        input_shape.AddDim(n);
        
        // Create diagonal tensor shape [batch_dim, min(m,n)]
        tensorflow::TensorShape diagonal_shape;
        diagonal_shape.AddDim(batch_dim);
        diagonal_shape.AddDim(std::min(m, n));
        
        size_t input_elements = input_shape.num_elements();
        size_t diagonal_elements = diagonal_shape.num_elements();
        size_t total_bytes_needed = (input_elements + diagonal_elements) * element_size;
        
        if (offset + total_bytes_needed > size) return 0;
        
        // Create input tensor
        tensorflow::Tensor input_tensor(dtype, input_shape);
        
        // Create diagonal tensor
        tensorflow::Tensor diagonal_tensor(dtype, diagonal_shape);
        
        // Fill tensors with fuzz data
        if (dtype == tensorflow::DT_FLOAT) {
            auto input_flat = input_tensor.flat<float>();
            auto diagonal_flat = diagonal_tensor.flat<float>();
            
            for (int i = 0; i < input_elements && offset + sizeof(float) <= size; ++i) {
                float val;
                memcpy(&val, data + offset, sizeof(float));
                input_flat(i) = val;
                offset += sizeof(float);
            }
            
            for (int i = 0; i < diagonal_elements && offset + sizeof(float) <= size; ++i) {
                float val;
                memcpy(&val, data + offset, sizeof(float));
                diagonal_flat(i) = val;
                offset += sizeof(float);
            }
        } else if (dtype == tensorflow::DT_DOUBLE) {
            auto input_flat = input_tensor.flat<double>();
            auto diagonal_flat = diagonal_tensor.flat<double>();
            
            for (int i = 0; i < input_elements && offset + sizeof(double) <= size; ++i) {
                double val;
                memcpy(&val, data + offset, sizeof(double));
                input_flat(i) = val;
                offset += sizeof(double);
            }
            
            for (int i = 0; i < diagonal_elements && offset + sizeof(double) <= size; ++i) {
                double val;
                memcpy(&val, data + offset, sizeof(double));
                diagonal_flat(i) = val;
                offset += sizeof(double);
            }
        } else { // DT_INT32
            auto input_flat = input_tensor.flat<int32_t>();
            auto diagonal_flat = diagonal_tensor.flat<int32_t>();
            
            for (int i = 0; i < input_elements && offset + sizeof(int32_t) <= size; ++i) {
                int32_t val;
                memcpy(&val, data + offset, sizeof(int32_t));
                input_flat(i) = val;
                offset += sizeof(int32_t);
            }
            
            for (int i = 0; i < diagonal_elements && offset + sizeof(int32_t) <= size; ++i) {
                int32_t val;
                memcpy(&val, data + offset, sizeof(int32_t));
                diagonal_flat(i) = val;
                offset += sizeof(int32_t);
            }
        }
        
        // Create a simple test using OpsTestBase-like approach
        tensorflow::Graph graph(tensorflow::OpRegistry::Global());
        
        tensorflow::Node* input_node;
        tensorflow::Node* diagonal_node;
        tensorflow::Node* matrix_set_diag_node;
        
        // Create placeholder nodes
        tensorflow::NodeDefBuilder input_builder("input", "Placeholder");
        input_builder.Attr("dtype", dtype);
        input_builder.Attr("shape", input_shape);
        tensorflow::NodeDef input_def;
        input_builder.Finalize(&input_def);
        tensorflow::Status status = graph.AddNode(input_def, &input_node);
        if (!status.ok()) return 0;
        
        tensorflow::NodeDefBuilder diagonal_builder("diagonal", "Placeholder");
        diagonal_builder.Attr("dtype", dtype);
        diagonal_builder.Attr("shape", diagonal_shape);
        tensorflow::NodeDef diagonal_def;
        diagonal_builder.Finalize(&diagonal_def);
        status = graph.AddNode(diagonal_def, &diagonal_node);
        if (!status.ok()) return 0;
        
        // Create MatrixSetDiag node
        tensorflow::NodeDefBuilder matrix_set_diag_builder("matrix_set_diag", "MatrixSetDiag");
        matrix_set_diag_builder.Input(input_node->name(), 0, dtype);
        matrix_set_diag_builder.Input(diagonal_node->name(), 0, dtype);
        matrix_set_diag_builder.Attr("T", dtype);
        tensorflow::NodeDef matrix_set_diag_def;
        matrix_set_diag_builder.Finalize(&matrix_set_diag_def);
        status = graph.AddNode(matrix_set_diag_def, &matrix_set_diag_node);
        if (!status.ok()) return 0;
        
        // Create session and run
        tensorflow::SessionOptions session_options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(session_options));
        if (!session) return 0;
        
        status = session->Create(graph.ToGraphDef());
        if (!status.ok()) return 0;
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"input:0", input_tensor},
            {"diagonal:0", diagonal_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {"matrix_set_diag:0"};
        
        status = session->Run(inputs, output_names, {}, &outputs);
        if (!status.ok()) return 0;
        
        // Verify output shape matches input shape
        if (outputs.size() == 1) {
            const tensorflow::Tensor& output = outputs[0];
            if (output.shape().dims() == input_shape.dims() &&
                output.dtype() == dtype) {
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