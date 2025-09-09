#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/public/session.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 8) return 0;
        
        // Extract dtype from fuzzer input
        uint32_t dtype_val = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;
        
        // Map to valid TensorFlow data types
        tensorflow::DataType dtype;
        switch (dtype_val % 10) {
            case 0: dtype = tensorflow::DT_FLOAT; break;
            case 1: dtype = tensorflow::DT_DOUBLE; break;
            case 2: dtype = tensorflow::DT_INT32; break;
            case 3: dtype = tensorflow::DT_INT64; break;
            case 4: dtype = tensorflow::DT_UINT8; break;
            case 5: dtype = tensorflow::DT_INT16; break;
            case 6: dtype = tensorflow::DT_INT8; break;
            case 7: dtype = tensorflow::DT_STRING; break;
            case 8: dtype = tensorflow::DT_BOOL; break;
            default: dtype = tensorflow::DT_FLOAT; break;
        }
        
        // Extract shape dimensions
        uint32_t num_dims = *reinterpret_cast<const uint32_t*>(data + offset) % 5; // Limit to 4 dimensions
        offset += 4;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        if (num_dims == 0 || offset + num_dims * 4 > size) {
            // Create placeholder with unknown shape
            auto placeholder = tensorflow::ops::Placeholder(root, dtype);
            
            // Try to get the output (this should not fail during graph construction)
            auto output = placeholder.output;
            
        } else {
            // Create shape from fuzzer input
            std::vector<int64_t> shape_dims;
            for (uint32_t i = 0; i < num_dims && offset + 4 <= size; ++i) {
                int32_t dim = *reinterpret_cast<const int32_t*>(data + offset);
                offset += 4;
                // Limit dimension size to reasonable values
                dim = dim % 1000;
                if (dim < 0) dim = -1; // -1 means unknown dimension
                shape_dims.push_back(dim);
            }
            
            tensorflow::PartialTensorShape shape(shape_dims);
            auto placeholder = tensorflow::ops::Placeholder(root, dtype, 
                tensorflow::ops::Placeholder::Shape(shape));
            
            // Try to get the output
            auto output = placeholder.output;
        }
        
        // Test with different name variations if there's remaining data
        if (offset < size) {
            std::string name = "test_placeholder";
            if (size - offset > 0) {
                // Use remaining data to create a name
                size_t name_len = std::min(size - offset, size_t(20));
                name = std::string(reinterpret_cast<const char*>(data + offset), name_len);
                // Replace null bytes and non-printable chars
                for (char& c : name) {
                    if (c == 0 || c < 32 || c > 126) c = 'x';
                }
            }
            
            tensorflow::Scope named_scope = root.WithOpName(name);
            auto named_placeholder = tensorflow::ops::Placeholder(named_scope, dtype);
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}