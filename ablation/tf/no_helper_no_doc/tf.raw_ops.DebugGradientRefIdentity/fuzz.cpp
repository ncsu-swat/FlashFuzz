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
        
        // Extract tensor dimensions
        int32_t dims = (data[offset] % 4) + 1;
        offset++;
        
        if (offset + dims * 4 > size) return 0;
        
        std::vector<int64_t> shape;
        for (int i = 0; i < dims; i++) {
            int32_t dim_size = *reinterpret_cast<const int32_t*>(data + offset) % 100 + 1;
            shape.push_back(std::abs(dim_size));
            offset += 4;
        }
        
        // Extract data type
        if (offset >= size) return 0;
        tensorflow::DataType dtype = static_cast<tensorflow::DataType>((data[offset] % 10) + 1);
        offset++;
        
        // Create tensor shape
        tensorflow::TensorShape tensor_shape(shape);
        
        // Calculate required data size
        size_t element_size = 0;
        switch (dtype) {
            case tensorflow::DT_FLOAT:
                element_size = sizeof(float);
                break;
            case tensorflow::DT_DOUBLE:
                element_size = sizeof(double);
                break;
            case tensorflow::DT_INT32:
                element_size = sizeof(int32_t);
                break;
            case tensorflow::DT_INT64:
                element_size = sizeof(int64_t);
                break;
            default:
                dtype = tensorflow::DT_FLOAT;
                element_size = sizeof(float);
                break;
        }
        
        size_t required_size = tensor_shape.num_elements() * element_size;
        if (offset + required_size > size) {
            // Use smaller tensor if not enough data
            tensor_shape = tensorflow::TensorShape({1});
            required_size = element_size;
        }
        
        // Create input tensor
        tensorflow::Tensor input_tensor(dtype, tensor_shape);
        
        // Fill tensor with fuzz data
        if (offset + required_size <= size) {
            std::memcpy(input_tensor.tensor_data().data(), data + offset, required_size);
        } else {
            // Fill with zeros if not enough data
            std::memset(input_tensor.tensor_data().data(), 0, required_size);
        }
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        // Create graph def
        tensorflow::GraphDef graph_def;
        
        // Add placeholder node
        tensorflow::NodeDef* placeholder = graph_def.add_node();
        placeholder->set_name("input");
        placeholder->set_op("Placeholder");
        (*placeholder->mutable_attr())["dtype"].set_type(dtype);
        
        // Add DebugGradientRefIdentity node
        tensorflow::NodeDef* debug_node = graph_def.add_node();
        debug_node->set_name("debug_gradient_ref_identity");
        debug_node->set_op("DebugGradientRefIdentity");
        debug_node->add_input("input");
        (*debug_node->mutable_attr())["T"].set_type(dtype);
        
        // Create session and run
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({{"input", input_tensor}}, 
                             {"debug_gradient_ref_identity"}, 
                             {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Verify output tensor has same shape and type as input
            const tensorflow::Tensor& output = outputs[0];
            if (output.dtype() == input_tensor.dtype() && 
                output.shape().IsSameSize(input_tensor.shape())) {
                // Operation succeeded
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