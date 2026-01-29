#include "fuzzer_utils.h"
#include <iostream>

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
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special.log1p operation
        torch::Tensor result = torch::special::log1p(input);
        
        // Access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            // Use sum() instead of item() to handle multi-element tensors
            volatile float val = result.sum().item<float>();
            (void)val;
        }
        
        // Test with out parameter version if we have more data
        if (offset + 4 < Size) {
            size_t offset2 = 0;
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset2);
            
            // Create output tensor with same shape
            torch::Tensor out_tensor = torch::empty_like(input2);
            
            // Test log1p_out
            try {
                torch::special::log1p_out(out_tensor, input2);
            } catch (...) {
                // Silently catch expected failures
            }
        }
        
        // Test with different dtypes
        if (Size > 1) {
            uint8_t dtype_selector = Data[Size - 1];
            
            // Test with float types (log1p works on floating point)
            torch::ScalarType dtypes[] = {
                torch::kFloat32,
                torch::kFloat64,
            };
            
            size_t dtype_idx = dtype_selector % 2;
            
            try {
                torch::Tensor input_cast = input.to(dtypes[dtype_idx]);
                torch::Tensor result_cast = torch::special::log1p(input_cast);
                
                if (result_cast.defined() && result_cast.numel() > 0) {
                    volatile double val = result_cast.sum().item<double>();
                    (void)val;
                }
            } catch (...) {
                // Silently catch dtype conversion failures
            }
        }
        
        // Test with specific shapes
        if (Size >= 2) {
            uint8_t dim1 = (Data[0] % 8) + 1;  // 1-8
            uint8_t dim2 = (Data[1] % 8) + 1;  // 1-8
            
            try {
                torch::Tensor shaped_input = torch::randn({dim1, dim2});
                torch::Tensor shaped_result = torch::special::log1p(shaped_input);
                
                if (shaped_result.defined() && shaped_result.numel() > 0) {
                    volatile float val = shaped_result.sum().item<float>();
                    (void)val;
                }
            } catch (...) {
                // Silently catch shape-related failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}