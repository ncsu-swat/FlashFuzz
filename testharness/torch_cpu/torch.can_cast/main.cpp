#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstdint>        // For uint64_t

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least 2 bytes for the data types
        if (Size < 2) {
            return 0;
        }
        
        // Parse the source and destination data types
        uint8_t src_dtype_selector = Data[offset++];
        uint8_t dst_dtype_selector = Data[offset++];
        
        // Get the actual ScalarType values
        torch::ScalarType src_dtype = fuzzer_utils::parseDataType(src_dtype_selector);
        torch::ScalarType dst_dtype = fuzzer_utils::parseDataType(dst_dtype_selector);
        
        // Test can_cast function - this is the main API being tested
        bool can_cast_result = torch::can_cast(src_dtype, dst_dtype);
        
        // Create a tensor with the source data type to test with actual data
        if (offset < Size) {
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Only attempt to cast if can_cast returned true
                if (can_cast_result) {
                    torch::Tensor cast_tensor = tensor.to(dst_dtype);
                }
            } catch (const std::exception& e) {
                // Expected for some dtype/shape combinations
            }
        }
        
        // Test with empty tensor
        try {
            if (can_cast_result) {
                torch::Tensor empty_tensor = torch::empty({0}, torch::TensorOptions().dtype(src_dtype));
                torch::Tensor cast_empty = empty_tensor.to(dst_dtype);
            }
        } catch (const std::exception& e) {
            // Some dtype combinations may fail even with empty tensors
        }
        
        // Test with scalar tensor
        try {
            if (can_cast_result) {
                torch::Tensor scalar_tensor = torch::scalar_tensor(1, torch::TensorOptions().dtype(src_dtype));
                torch::Tensor cast_scalar = scalar_tensor.to(dst_dtype);
            }
        } catch (const std::exception& e) {
            // Some dtype combinations may fail
        }
        
        // Test with tensor containing extreme values
        try {
            if (can_cast_result) {
                torch::Tensor extreme_tensor;
                
                if (src_dtype == torch::kFloat32 || src_dtype == torch::kFloat64) {
                    extreme_tensor = torch::tensor({std::numeric_limits<float>::max(), 
                                                   std::numeric_limits<float>::min(),
                                                   std::numeric_limits<float>::infinity(),
                                                   -std::numeric_limits<float>::infinity(),
                                                   std::numeric_limits<float>::quiet_NaN()}, 
                                                   torch::TensorOptions().dtype(src_dtype));
                } else if (src_dtype == torch::kInt32 || src_dtype == torch::kInt64) {
                    extreme_tensor = torch::tensor({std::numeric_limits<int32_t>::max(), 
                                                   std::numeric_limits<int32_t>::min()}, 
                                                   torch::TensorOptions().dtype(src_dtype));
                } else if (src_dtype == torch::kBool) {
                    extreme_tensor = torch::tensor({true, false}, 
                                                   torch::TensorOptions().dtype(src_dtype));
                } else {
                    extreme_tensor = torch::ones({2}, torch::TensorOptions().dtype(src_dtype));
                }
                
                torch::Tensor cast_extreme = extreme_tensor.to(dst_dtype);
            }
        } catch (const std::exception& e) {
            // Expected for some dtype combinations with extreme values
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}