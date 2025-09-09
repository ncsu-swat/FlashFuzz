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
        
        // Extract batch size (1-4 bytes)
        uint32_t batch_size = (data[offset] % 10) + 1;
        offset += 1;
        
        // Extract dimensions (1-3 additional dimensions)
        uint32_t dim1 = (data[offset] % 10) + 1;
        offset += 1;
        uint32_t dim2 = (data[offset] % 10) + 1;
        offset += 1;
        
        // Extract data type
        tensorflow::DataType dtype;
        uint8_t dtype_choice = data[offset] % 4;
        offset += 1;
        
        switch (dtype_choice) {
            case 0: dtype = tensorflow::DT_INT32; break;
            case 1: dtype = tensorflow::DT_FLOAT; break;
            case 2: dtype = tensorflow::DT_INT64; break;
            default: dtype = tensorflow::DT_DOUBLE; break;
        }
        
        // Create input tensor shape
        tensorflow::TensorShape input_shape({static_cast<int64_t>(batch_size), 
                                           static_cast<int64_t>(dim1), 
                                           static_cast<int64_t>(dim2)});
        
        // Create input tensor
        tensorflow::Tensor input_tensor(dtype, input_shape);
        
        // Fill tensor with fuzz data
        size_t tensor_bytes = input_tensor.TotalBytes();
        size_t available_bytes = size - offset;
        
        if (available_bytes > 0) {
            void* tensor_data = input_tensor.data();
            size_t copy_bytes = std::min(tensor_bytes, available_bytes);
            std::memcpy(tensor_data, data + offset, copy_bytes);
            
            // Fill remaining bytes with zeros if needed
            if (copy_bytes < tensor_bytes) {
                std::memset(static_cast<uint8_t*>(tensor_data) + copy_bytes, 0, 
                           tensor_bytes - copy_bytes);
            }
        }
        
        // Create method tensor (farmhash64)
        tensorflow::Tensor method_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        method_tensor.scalar<tensorflow::tstring>()() = "farmhash64";
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Build graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* data_node = graph_def.add_node();
        data_node->set_name("input_data");
        data_node->set_op("Placeholder");
        (*data_node->mutable_attr())["dtype"].set_type(dtype);
        
        tensorflow::NodeDef* method_node = graph_def.add_node();
        method_node->set_name("input_method");
        method_node->set_op("Placeholder");
        (*method_node->mutable_attr())["dtype"].set_type(tensorflow::DT_STRING);
        
        tensorflow::NodeDef* fingerprint_node = graph_def.add_node();
        fingerprint_node->set_name("fingerprint");
        fingerprint_node->set_op("Fingerprint");
        fingerprint_node->add_input("input_data");
        fingerprint_node->add_input("input_method");
        
        // Create session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            {"input_data", input_tensor},
            {"input_method", method_tensor}
        };
        
        std::vector<tensorflow::Tensor> outputs;
        std::vector<std::string> output_names = {"fingerprint"};
        
        status = session->Run(inputs, output_names, {}, &outputs);
        if (status.ok() && !outputs.empty()) {
            // Verify output shape and type
            const tensorflow::Tensor& output = outputs[0];
            if (output.dtype() == tensorflow::DT_UINT8 && 
                output.dims() == 2 && 
                output.dim_size(0) == batch_size &&
                output.dim_size(1) == 8) {
                // Output is valid
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