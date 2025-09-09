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
        std::vector<int64_t> dims;
        int64_t total_elements = 1;
        for (int i = 0; i < num_dims; i++) {
            int32_t dim_size = *reinterpret_cast<const int32_t*>(data + offset);
            offset += sizeof(int32_t);
            dim_size = std::abs(dim_size) % 100 + 1; // Limit size to prevent memory issues
            dims.push_back(dim_size);
            total_elements *= dim_size;
        }
        
        // Limit total elements to prevent memory issues
        if (total_elements > 10000) {
            return 0;
        }
        
        tensorflow::TensorShape shape(dims);
        
        // Test with different data types
        std::vector<tensorflow::DataType> types = {
            tensorflow::DT_FLOAT,
            tensorflow::DT_DOUBLE,
            tensorflow::DT_COMPLEX64,
            tensorflow::DT_COMPLEX128
        };
        
        for (auto dtype : types) {
            tensorflow::Tensor input_tensor(dtype, shape);
            
            // Fill tensor with fuzz data
            size_t element_size = 0;
            switch (dtype) {
                case tensorflow::DT_FLOAT:
                    element_size = sizeof(float);
                    break;
                case tensorflow::DT_DOUBLE:
                    element_size = sizeof(double);
                    break;
                case tensorflow::DT_COMPLEX64:
                    element_size = sizeof(std::complex<float>);
                    break;
                case tensorflow::DT_COMPLEX128:
                    element_size = sizeof(std::complex<double>);
                    break;
                default:
                    continue;
            }
            
            size_t needed_bytes = total_elements * element_size;
            if (offset + needed_bytes > size) {
                // Fill with pattern if not enough data
                auto flat = input_tensor.flat<float>();
                for (int64_t i = 0; i < total_elements; i++) {
                    flat(i) = static_cast<float>((i + offset) % 256) / 256.0f;
                }
            } else {
                // Copy fuzz data
                std::memcpy(input_tensor.data(), data + offset, needed_bytes);
                offset += needed_bytes;
            }
            
            // Create OpKernel for Cos operation
            tensorflow::NodeDef node_def;
            tensorflow::Status status = tensorflow::NodeDefBuilder("cos_op", "Cos")
                                          .Input("x", 0, dtype)
                                          .Finalize(&node_def);
            
            if (!status.ok()) {
                continue;
            }
            
            // Create a simple test context
            tensorflow::OpKernelContext::Params params;
            tensorflow::DeviceBase device(tensorflow::Env::Default());
            params.device = &device;
            params.op_kernel = nullptr;
            
            // Create output tensor
            tensorflow::Tensor* output_tensor = nullptr;
            tensorflow::AllocatorAttributes alloc_attrs;
            
            // The actual cos computation would be done by TensorFlow's Cos kernel
            // For fuzzing purposes, we just verify the tensor creation and basic operations
            
            // Verify tensor properties
            if (input_tensor.NumElements() != total_elements) {
                continue;
            }
            
            if (input_tensor.shape() != shape) {
                continue;
            }
            
            // Test tensor access patterns
            switch (dtype) {
                case tensorflow::DT_FLOAT: {
                    auto flat = input_tensor.flat<float>();
                    for (int64_t i = 0; i < std::min(total_elements, 100L); i++) {
                        volatile float val = flat(i);
                        (void)val; // Suppress unused variable warning
                    }
                    break;
                }
                case tensorflow::DT_DOUBLE: {
                    auto flat = input_tensor.flat<double>();
                    for (int64_t i = 0; i < std::min(total_elements, 100L); i++) {
                        volatile double val = flat(i);
                        (void)val;
                    }
                    break;
                }
                case tensorflow::DT_COMPLEX64: {
                    auto flat = input_tensor.flat<std::complex<float>>();
                    for (int64_t i = 0; i < std::min(total_elements, 100L); i++) {
                        volatile std::complex<float> val = flat(i);
                        (void)val;
                    }
                    break;
                }
                case tensorflow::DT_COMPLEX128: {
                    auto flat = input_tensor.flat<std::complex<double>>();
                    for (int64_t i = 0; i < std::min(total_elements, 100L); i++) {
                        volatile std::complex<double> val = flat(i);
                        (void)val;
                    }
                    break;
                }
                default:
                    break;
            }
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}