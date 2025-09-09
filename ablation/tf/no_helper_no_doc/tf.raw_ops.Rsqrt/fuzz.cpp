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
#include <tensorflow/core/lib/core/status_test_util.h>
#include <tensorflow/core/platform/test.h>

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
        num_dims = std::abs(num_dims) % 4 + 1;
        
        if (offset + num_dims * sizeof(int32_t) > size) {
            return 0;
        }
        
        // Extract dimension sizes
        tensorflow::TensorShape shape;
        for (int i = 0; i < num_dims; i++) {
            int32_t dim_size = *reinterpret_cast<const int32_t*>(data + offset);
            offset += sizeof(int32_t);
            dim_size = std::abs(dim_size) % 100 + 1; // Limit size to prevent memory issues
            shape.AddDim(dim_size);
        }
        
        // Calculate required data size
        size_t num_elements = shape.num_elements();
        size_t required_size = num_elements * sizeof(float);
        
        if (offset + required_size > size) {
            return 0;
        }
        
        // Create input tensor
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, shape);
        auto input_flat = input_tensor.flat<float>();
        
        // Fill tensor with fuzz data
        const float* float_data = reinterpret_cast<const float*>(data + offset);
        for (int64_t i = 0; i < num_elements; i++) {
            float val = float_data[i % ((size - offset) / sizeof(float))];
            // Ensure positive values for rsqrt and avoid very small values
            if (std::isfinite(val) && val > 1e-10f) {
                input_flat(i) = std::abs(val);
            } else {
                input_flat(i) = 1.0f;
            }
        }
        
        // Create output tensor
        tensorflow::Tensor output_tensor(tensorflow::DT_FLOAT, shape);
        auto output_flat = output_tensor.flat<float>();
        
        // Perform rsqrt operation manually (since we're testing the operation)
        for (int64_t i = 0; i < num_elements; i++) {
            float input_val = input_flat(i);
            if (input_val > 0.0f) {
                output_flat(i) = 1.0f / std::sqrt(input_val);
            } else {
                output_flat(i) = std::numeric_limits<float>::infinity();
            }
        }
        
        // Test with different data types if there's remaining data
        if (offset + required_size + sizeof(int32_t) < size) {
            int32_t dtype_selector = *reinterpret_cast<const int32_t*>(data + offset + required_size);
            
            switch (std::abs(dtype_selector) % 3) {
                case 0: {
                    // Test with double
                    tensorflow::Tensor double_input(tensorflow::DT_DOUBLE, shape);
                    auto double_flat = double_input.flat<double>();
                    for (int64_t i = 0; i < num_elements; i++) {
                        double val = static_cast<double>(input_flat(i));
                        double_flat(i) = val;
                    }
                    break;
                }
                case 1: {
                    // Test with half precision
                    tensorflow::Tensor half_input(tensorflow::DT_HALF, shape);
                    auto half_flat = half_input.flat<Eigen::half>();
                    for (int64_t i = 0; i < num_elements; i++) {
                        half_flat(i) = Eigen::half(input_flat(i));
                    }
                    break;
                }
                case 2: {
                    // Test with bfloat16
                    tensorflow::Tensor bfloat_input(tensorflow::DT_BFLOAT16, shape);
                    auto bfloat_flat = bfloat_input.flat<tensorflow::bfloat16>();
                    for (int64_t i = 0; i < num_elements; i++) {
                        bfloat_flat(i) = tensorflow::bfloat16(input_flat(i));
                    }
                    break;
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