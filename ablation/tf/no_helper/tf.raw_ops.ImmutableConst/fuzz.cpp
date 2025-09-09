#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/cc/framework/scope.h>
#include <fstream>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dtype (limit to common types)
        uint8_t dtype_val = data[offset++] % 4;
        tensorflow::DataType dtype;
        switch (dtype_val) {
            case 0: dtype = tensorflow::DT_FLOAT; break;
            case 1: dtype = tensorflow::DT_INT32; break;
            case 2: dtype = tensorflow::DT_DOUBLE; break;
            case 3: dtype = tensorflow::DT_INT64; break;
            default: dtype = tensorflow::DT_FLOAT; break;
        }
        
        // Extract shape dimensions (limit to reasonable values)
        uint8_t num_dims = (data[offset++] % 4) + 1; // 1-4 dimensions
        std::vector<int64_t> shape_dims;
        for (int i = 0; i < num_dims && offset < size; i++) {
            int64_t dim = (data[offset++] % 10) + 1; // 1-10 size per dimension
            shape_dims.push_back(dim);
        }
        
        if (offset >= size) return 0;
        
        // Create memory region name from remaining data
        std::string memory_region_name = "/tmp/tf_test_region_";
        for (size_t i = offset; i < size && i < offset + 8; i++) {
            memory_region_name += std::to_string(data[i] % 10);
        }
        
        // Create a temporary file with some data
        std::string temp_file = memory_region_name + ".dat";
        std::ofstream file(temp_file, std::ios::binary);
        if (file.is_open()) {
            // Calculate total elements
            int64_t total_elements = 1;
            for (auto dim : shape_dims) {
                total_elements *= dim;
            }
            
            // Write some dummy data based on dtype
            size_t element_size = 4; // default for float/int32
            if (dtype == tensorflow::DT_DOUBLE || dtype == tensorflow::DT_INT64) {
                element_size = 8;
            }
            
            std::vector<uint8_t> dummy_data(total_elements * element_size, 0);
            for (size_t i = 0; i < dummy_data.size() && i + offset < size; i++) {
                dummy_data[i] = data[(offset + i) % size];
            }
            
            file.write(reinterpret_cast<const char*>(dummy_data.data()), dummy_data.size());
            file.close();
            
            // Create TensorFlow scope and session
            tensorflow::Scope root = tensorflow::Scope::NewRootScope();
            
            // Create TensorShape
            tensorflow::TensorShape tensor_shape;
            for (auto dim : shape_dims) {
                tensor_shape.AddDim(dim);
            }
            
            // Create ImmutableConst operation
            auto immutable_const = tensorflow::ops::ImmutableConst(
                root.WithOpName("test_immutable_const"),
                tensor_shape,
                dtype,
                temp_file
            );
            
            // Create session and run
            tensorflow::ClientSession session(root);
            std::vector<tensorflow::Tensor> outputs;
            
            auto status = session.Run({immutable_const}, &outputs);
            
            // Clean up temporary file
            std::remove(temp_file.c_str());
            
            if (status.ok() && !outputs.empty()) {
                // Verify output tensor properties
                const auto& output_tensor = outputs[0];
                if (output_tensor.dtype() == dtype && 
                    output_tensor.shape().dims() == tensor_shape.dims()) {
                    // Success case - tensor created successfully
                }
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}