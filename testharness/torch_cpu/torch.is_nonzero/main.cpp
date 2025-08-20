#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply is_nonzero operation
        bool result = torch::is_nonzero(tensor);
        
        // Try to use the result to prevent optimization
        if (result) {
            volatile int dummy = 1;
        }
        
        // Try with different tensor types and shapes
        if (offset + 2 < Size) {
            // Create another tensor with different properties
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            bool result2 = torch::is_nonzero(tensor2);
            
            if (result2) {
                volatile int dummy = 1;
            }
        }
        
        // Test with scalar tensors
        if (offset + 1 < Size) {
            // Create a scalar tensor with a single value
            torch::Tensor scalar_tensor = torch::tensor(static_cast<int>(Data[offset]));
            bool scalar_result = torch::is_nonzero(scalar_tensor);
            
            if (scalar_result) {
                volatile int dummy = 1;
            }
        }
        
        // Test with boolean tensors
        if (offset + 1 < Size) {
            bool bool_value = Data[offset] % 2 == 0;
            torch::Tensor bool_tensor = torch::tensor(bool_value);
            bool bool_result = torch::is_nonzero(bool_tensor);
            
            if (bool_result) {
                volatile int dummy = 1;
            }
        }
        
        // Test with zero-sized tensors if possible
        if (offset + 1 < Size) {
            std::vector<int64_t> empty_shape = {0};
            torch::Tensor empty_tensor = torch::empty(empty_shape);
            
            try {
                bool empty_result = torch::is_nonzero(empty_tensor);
                if (empty_result) {
                    volatile int dummy = 1;
                }
            } catch (const std::exception &) {
                // Expected exception for empty tensor
            }
        }
        
        // Test with tensors containing NaN or Inf
        if (offset + 1 < Size) {
            torch::Tensor nan_tensor = torch::tensor(std::numeric_limits<float>::quiet_NaN());
            torch::Tensor inf_tensor = torch::tensor(std::numeric_limits<float>::infinity());
            
            try {
                bool nan_result = torch::is_nonzero(nan_tensor);
                if (nan_result) {
                    volatile int dummy = 1;
                }
            } catch (const std::exception &) {
                // Handle potential exception
            }
            
            try {
                bool inf_result = torch::is_nonzero(inf_tensor);
                if (inf_result) {
                    volatile int dummy = 1;
                }
            } catch (const std::exception &) {
                // Handle potential exception
            }
        }
        
        // Test with multi-element tensors (should throw exception)
        if (offset + 1 < Size) {
            torch::Tensor multi_tensor = torch::ones({2, 2});
            
            try {
                bool multi_result = torch::is_nonzero(multi_tensor);
                if (multi_result) {
                    volatile int dummy = 1;
                }
            } catch (const std::exception &) {
                // Expected exception for multi-element tensor
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}