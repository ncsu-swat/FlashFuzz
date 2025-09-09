#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/kernel_def_builder.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions and data type info from fuzz input
        uint8_t rank = data[offset++] % 4 + 1; // 1-4 dimensions
        uint8_t dtype_idx = data[offset++] % 3; // 0=float, 1=int32, 2=bool
        
        tensorflow::DataType dtype;
        size_t element_size;
        switch (dtype_idx) {
            case 0: dtype = tensorflow::DT_FLOAT; element_size = sizeof(float); break;
            case 1: dtype = tensorflow::DT_INT32; element_size = sizeof(int32_t); break;
            case 2: dtype = tensorflow::DT_BOOL; element_size = sizeof(bool); break;
            default: dtype = tensorflow::DT_FLOAT; element_size = sizeof(float); break;
        }
        
        // Create tensor shapes
        std::vector<int64_t> dims;
        int64_t total_elements = 1;
        for (int i = 0; i < rank && offset < size; i++) {
            int64_t dim = (data[offset++] % 8) + 1; // 1-8 elements per dimension
            dims.push_back(dim);
            total_elements *= dim;
            if (total_elements > 1000) { // Limit size to prevent excessive memory usage
                total_elements = 1000;
                break;
            }
        }
        
        tensorflow::TensorShape shape(dims);
        
        // Calculate required data size
        size_t required_condition_size = total_elements * sizeof(bool);
        size_t required_data_size = total_elements * element_size;
        size_t total_required = required_condition_size + 2 * required_data_size;
        
        if (offset + total_required > size) return 0;
        
        // Create condition tensor (bool)
        tensorflow::Tensor condition_tensor(tensorflow::DT_BOOL, shape);
        auto condition_flat = condition_tensor.flat<bool>();
        for (int64_t i = 0; i < total_elements && offset < size; i++) {
            condition_flat(i) = (data[offset++] % 2) == 1;
        }
        
        // Create t tensor (then branch)
        tensorflow::Tensor t_tensor(dtype, shape);
        if (dtype == tensorflow::DT_FLOAT) {
            auto t_flat = t_tensor.flat<float>();
            for (int64_t i = 0; i < total_elements && offset + sizeof(float) <= size; i++) {
                float val;
                memcpy(&val, &data[offset], sizeof(float));
                t_flat(i) = val;
                offset += sizeof(float);
            }
        } else if (dtype == tensorflow::DT_INT32) {
            auto t_flat = t_tensor.flat<int32_t>();
            for (int64_t i = 0; i < total_elements && offset + sizeof(int32_t) <= size; i++) {
                int32_t val;
                memcpy(&val, &data[offset], sizeof(int32_t));
                t_flat(i) = val;
                offset += sizeof(int32_t);
            }
        } else if (dtype == tensorflow::DT_BOOL) {
            auto t_flat = t_tensor.flat<bool>();
            for (int64_t i = 0; i < total_elements && offset < size; i++) {
                t_flat(i) = (data[offset++] % 2) == 1;
            }
        }
        
        // Create e tensor (else branch)
        tensorflow::Tensor e_tensor(dtype, shape);
        if (dtype == tensorflow::DT_FLOAT) {
            auto e_flat = e_tensor.flat<float>();
            for (int64_t i = 0; i < total_elements && offset + sizeof(float) <= size; i++) {
                float val;
                memcpy(&val, &data[offset], sizeof(float));
                e_flat(i) = val;
                offset += sizeof(float);
            }
        } else if (dtype == tensorflow::DT_INT32) {
            auto e_flat = e_tensor.flat<int32_t>();
            for (int64_t i = 0; i < total_elements && offset + sizeof(int32_t) <= size; i++) {
                int32_t val;
                memcpy(&val, &data[offset], sizeof(int32_t));
                e_flat(i) = val;
                offset += sizeof(int32_t);
            }
        } else if (dtype == tensorflow::DT_BOOL) {
            auto e_flat = e_tensor.flat<bool>();
            for (int64_t i = 0; i < total_elements && offset < size; i++) {
                e_flat(i) = (data[offset++] % 2) == 1;
            }
        }
        
        // Create output tensor
        tensorflow::Tensor output_tensor(dtype, shape);
        
        // Perform SelectV2 operation manually (element-wise selection)
        if (dtype == tensorflow::DT_FLOAT) {
            auto condition_flat = condition_tensor.flat<bool>();
            auto t_flat = t_tensor.flat<float>();
            auto e_flat = e_tensor.flat<float>();
            auto output_flat = output_tensor.flat<float>();
            
            for (int64_t i = 0; i < total_elements; i++) {
                output_flat(i) = condition_flat(i) ? t_flat(i) : e_flat(i);
            }
        } else if (dtype == tensorflow::DT_INT32) {
            auto condition_flat = condition_tensor.flat<bool>();
            auto t_flat = t_tensor.flat<int32_t>();
            auto e_flat = e_tensor.flat<int32_t>();
            auto output_flat = output_tensor.flat<int32_t>();
            
            for (int64_t i = 0; i < total_elements; i++) {
                output_flat(i) = condition_flat(i) ? t_flat(i) : e_flat(i);
            }
        } else if (dtype == tensorflow::DT_BOOL) {
            auto condition_flat = condition_tensor.flat<bool>();
            auto t_flat = t_tensor.flat<bool>();
            auto e_flat = e_tensor.flat<bool>();
            auto output_flat = output_tensor.flat<bool>();
            
            for (int64_t i = 0; i < total_elements; i++) {
                output_flat(i) = condition_flat(i) ? t_flat(i) : e_flat(i);
            }
        }
        
        // Test broadcasting scenarios if there's remaining data
        if (offset < size && size - offset > 8) {
            // Create scalar condition with vector inputs
            tensorflow::TensorShape scalar_shape({});
            tensorflow::TensorShape vector_shape({std::min((int64_t)((size - offset) / (2 * element_size)), (int64_t)10)});
            
            tensorflow::Tensor scalar_condition(tensorflow::DT_BOOL, scalar_shape);
            scalar_condition.scalar<bool>()() = (data[offset++] % 2) == 1;
            
            tensorflow::Tensor vector_t(dtype, vector_shape);
            tensorflow::Tensor vector_e(dtype, vector_shape);
            tensorflow::Tensor vector_output(dtype, vector_shape);
            
            int64_t vector_elements = vector_shape.num_elements();
            
            if (dtype == tensorflow::DT_FLOAT && offset + 2 * vector_elements * sizeof(float) <= size) {
                auto t_flat = vector_t.flat<float>();
                auto e_flat = vector_e.flat<float>();
                auto output_flat = vector_output.flat<float>();
                bool condition_val = scalar_condition.scalar<bool>()();
                
                for (int64_t i = 0; i < vector_elements; i++) {
                    float t_val, e_val;
                    memcpy(&t_val, &data[offset], sizeof(float));
                    offset += sizeof(float);
                    memcpy(&e_val, &data[offset], sizeof(float));
                    offset += sizeof(float);
                    
                    t_flat(i) = t_val;
                    e_flat(i) = e_val;
                    output_flat(i) = condition_val ? t_val : e_val;
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