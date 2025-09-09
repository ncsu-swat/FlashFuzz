#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/common_runtime/kernel_benchmark_testlib.h>
#include <tensorflow/core/framework/fake_input.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/types.pb.h>
#include <tensorflow/core/kernels/ops_testutil.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < sizeof(int32_t) * 2) {
            return 0;
        }
        
        // Extract tensor dimensions
        int32_t num_dims = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        // Limit dimensions to reasonable range
        num_dims = std::max(1, std::min(num_dims, 4));
        
        if (offset + num_dims * sizeof(int32_t) > size) {
            return 0;
        }
        
        // Extract dimension sizes
        std::vector<int64_t> dims;
        int64_t total_elements = 1;
        for (int i = 0; i < num_dims; i++) {
            int32_t dim_size = *reinterpret_cast<const int32_t*>(data + offset);
            offset += sizeof(int32_t);
            dim_size = std::max(1, std::min(dim_size, 100)); // Limit size
            dims.push_back(dim_size);
            total_elements *= dim_size;
        }
        
        // Limit total elements to prevent excessive memory usage
        if (total_elements > 10000) {
            return 0;
        }
        
        tensorflow::TensorShape shape(dims);
        
        // Determine data type
        tensorflow::DataType dtype = tensorflow::DT_FLOAT;
        if (offset < size) {
            uint8_t type_selector = data[offset++];
            switch (type_selector % 4) {
                case 0: dtype = tensorflow::DT_FLOAT; break;
                case 1: dtype = tensorflow::DT_DOUBLE; break;
                case 2: dtype = tensorflow::DT_HALF; break;
                case 3: dtype = tensorflow::DT_BFLOAT16; break;
            }
        }
        
        // Create input tensor
        tensorflow::Tensor input_tensor(dtype, shape);
        
        // Fill tensor with data
        size_t element_size = 0;
        switch (dtype) {
            case tensorflow::DT_FLOAT:
                element_size = sizeof(float);
                break;
            case tensorflow::DT_DOUBLE:
                element_size = sizeof(double);
                break;
            case tensorflow::DT_HALF:
                element_size = sizeof(tensorflow::bfloat16);
                break;
            case tensorflow::DT_BFLOAT16:
                element_size = sizeof(tensorflow::bfloat16);
                break;
        }
        
        size_t required_data = total_elements * element_size;
        if (offset + required_data > size) {
            // Fill with zeros if not enough data
            memset(input_tensor.data(), 0, required_data);
        } else {
            // Copy available data
            size_t available_data = std::min(required_data, size - offset);
            memcpy(input_tensor.data(), data + offset, available_data);
            if (available_data < required_data) {
                memset(static_cast<char*>(input_tensor.data()) + available_data, 0, 
                       required_data - available_data);
            }
        }
        
        // Create session and graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* input_node = graph_def.add_node();
        input_node->set_name("input");
        input_node->set_op("Placeholder");
        input_node->mutable_attr()->insert({"dtype", tensorflow::AttrValue()});
        input_node->mutable_attr()->at("dtype").set_type(dtype);
        
        tensorflow::NodeDef* exp_node = graph_def.add_node();
        exp_node->set_name("exp");
        exp_node->set_op("Exp");
        exp_node->add_input("input");
        exp_node->mutable_attr()->insert({"T", tensorflow::AttrValue()});
        exp_node->mutable_attr()->at("T").set_type(dtype);
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        if (session == nullptr) {
            return 0;
        }
        
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({{"input", input_tensor}}, {"exp"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Successfully computed exp operation
            const tensorflow::Tensor& output = outputs[0];
            // Verify output shape matches input shape
            if (output.shape().dims() == input_tensor.shape().dims()) {
                for (int i = 0; i < output.shape().dims(); i++) {
                    if (output.shape().dim_size(i) != input_tensor.shape().dim_size(i)) {
                        return -1;
                    }
                }
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