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
        int32_t precision = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        bool scientific = (data[offset] % 2) == 1;
        offset += 1;
        
        bool shortest = (data[offset] % 2) == 1;
        offset += 1;
        
        int32_t width = *reinterpret_cast<const int32_t*>(data + offset);
        offset += sizeof(int32_t);
        
        char fill_char = static_cast<char>(data[offset]);
        offset += 1;
        
        // Determine input tensor type and create tensor
        tensorflow::DataType input_type = static_cast<tensorflow::DataType>(data[offset] % 19 + 1); // DT_FLOAT to DT_COMPLEX128
        offset += 1;
        
        // Create input tensor shape
        int32_t num_dims = (data[offset] % 4) + 1; // 1-4 dimensions
        offset += 1;
        
        if (offset + num_dims * sizeof(int32_t) > size) return 0;
        
        tensorflow::TensorShape shape;
        int32_t total_elements = 1;
        for (int i = 0; i < num_dims; i++) {
            int32_t dim_size = std::abs(*reinterpret_cast<const int32_t*>(data + offset)) % 10 + 1;
            offset += sizeof(int32_t);
            shape.AddDim(dim_size);
            total_elements *= dim_size;
        }
        
        if (total_elements > 1000) total_elements = 1000; // Limit size
        
        // Create input tensor based on type
        tensorflow::Tensor input_tensor;
        
        switch (input_type) {
            case tensorflow::DT_FLOAT: {
                input_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, shape);
                auto flat = input_tensor.flat<float>();
                for (int i = 0; i < std::min(total_elements, static_cast<int32_t>((size - offset) / sizeof(float))); i++) {
                    if (offset + sizeof(float) <= size) {
                        flat(i) = *reinterpret_cast<const float*>(data + offset);
                        offset += sizeof(float);
                    }
                }
                break;
            }
            case tensorflow::DT_DOUBLE: {
                input_tensor = tensorflow::Tensor(tensorflow::DT_DOUBLE, shape);
                auto flat = input_tensor.flat<double>();
                for (int i = 0; i < std::min(total_elements, static_cast<int32_t>((size - offset) / sizeof(double))); i++) {
                    if (offset + sizeof(double) <= size) {
                        flat(i) = *reinterpret_cast<const double*>(data + offset);
                        offset += sizeof(double);
                    }
                }
                break;
            }
            case tensorflow::DT_INT32: {
                input_tensor = tensorflow::Tensor(tensorflow::DT_INT32, shape);
                auto flat = input_tensor.flat<int32_t>();
                for (int i = 0; i < std::min(total_elements, static_cast<int32_t>((size - offset) / sizeof(int32_t))); i++) {
                    if (offset + sizeof(int32_t) <= size) {
                        flat(i) = *reinterpret_cast<const int32_t*>(data + offset);
                        offset += sizeof(int32_t);
                    }
                }
                break;
            }
            case tensorflow::DT_INT64: {
                input_tensor = tensorflow::Tensor(tensorflow::DT_INT64, shape);
                auto flat = input_tensor.flat<int64_t>();
                for (int i = 0; i < std::min(total_elements, static_cast<int32_t>((size - offset) / sizeof(int64_t))); i++) {
                    if (offset + sizeof(int64_t) <= size) {
                        flat(i) = *reinterpret_cast<const int64_t*>(data + offset);
                        offset += sizeof(int64_t);
                    }
                }
                break;
            }
            case tensorflow::DT_BOOL: {
                input_tensor = tensorflow::Tensor(tensorflow::DT_BOOL, shape);
                auto flat = input_tensor.flat<bool>();
                for (int i = 0; i < std::min(total_elements, static_cast<int32_t>(size - offset)); i++) {
                    if (offset < size) {
                        flat(i) = (data[offset] % 2) == 1;
                        offset += 1;
                    }
                }
                break;
            }
            default:
                // Default to float for unsupported types
                input_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, shape);
                break;
        }
        
        // Create session and graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* input_node = graph_def.add_node();
        input_node->set_name("input");
        input_node->set_op("Placeholder");
        (*input_node->mutable_attr())["dtype"].set_type(input_tensor.dtype());
        
        tensorflow::NodeDef* as_string_node = graph_def.add_node();
        as_string_node->set_name("as_string");
        as_string_node->set_op("AsString");
        as_string_node->add_input("input");
        (*as_string_node->mutable_attr())["T"].set_type(input_tensor.dtype());
        (*as_string_node->mutable_attr())["precision"].set_i(precision);
        (*as_string_node->mutable_attr())["scientific"].set_b(scientific);
        (*as_string_node->mutable_attr())["shortest"].set_b(shortest);
        (*as_string_node->mutable_attr())["width"].set_i(width);
        (*as_string_node->mutable_attr())["fill"].set_s(std::string(1, fill_char));
        
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
        tensorflow::Status status = session->Create(graph_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Run the operation
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run({{"input", input_tensor}}, {"as_string"}, {}, &outputs);
        
        if (status.ok() && !outputs.empty()) {
            // Successfully executed AsString operation
            const tensorflow::Tensor& output = outputs[0];
            if (output.dtype() == tensorflow::DT_STRING) {
                auto output_flat = output.flat<tensorflow::tstring>();
                // Access the string results to ensure they're valid
                for (int i = 0; i < std::min(10, static_cast<int>(output_flat.size())); i++) {
                    std::string result = output_flat(i);
                    // Just access the string to trigger any potential issues
                    volatile size_t len = result.length();
                    (void)len;
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