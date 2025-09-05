#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 3 bytes for control flags and minimal tensor creation
        if (Size < 3) {
            return 0;
        }

        // Parse control flags
        uint8_t accumulate_flag = Data[offset++];
        bool accumulate = accumulate_flag & 0x01;
        
        // Create input tensor
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If we can't create the first tensor, try with a default one
            input = torch::randn({5, 3});
        }
        
        // Create indices tensor
        torch::Tensor index;
        try {
            index = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure indices are of integer type
            if (!index.dtype().isIntegral(false)) {
                index = index.to(torch::kLong);
            }
        } catch (const std::exception& e) {
            // Create default indices if parsing fails
            index = torch::randint(0, input.numel(), {std::min(static_cast<int64_t>(3), input.numel())});
        }
        
        // Create values tensor
        torch::Tensor values;
        try {
            values = fuzzer_utils::createTensor(Data, Size, offset);
            // Convert values to match input dtype if needed
            if (values.dtype() != input.dtype()) {
                values = values.to(input.dtype());
            }
        } catch (const std::exception& e) {
            // Create values matching the size of indices
            values = torch::randn({index.numel()}, torch::TensorOptions().dtype(input.dtype()));
        }
        
        // Flatten index tensor to 1D if needed
        if (index.dim() != 1) {
            index = index.flatten();
        }
        
        // Ensure values tensor is 1D and matches index size
        if (values.dim() != 1) {
            values = values.flatten();
        }
        
        // Resize values if needed to match index size
        if (values.numel() != index.numel()) {
            if (values.numel() > index.numel()) {
                values = values.slice(0, 0, index.numel());
            } else if (values.numel() > 0) {
                // Repeat values to match index size
                int64_t repeat_times = (index.numel() + values.numel() - 1) / values.numel();
                values = values.repeat({repeat_times}).slice(0, 0, index.numel());
            } else {
                // Create new values if empty
                values = torch::zeros({index.numel()}, torch::TensorOptions().dtype(input.dtype()));
            }
        }
        
        // Clone input to avoid modifying the original (for potential comparison)
        torch::Tensor result = input.clone();
        
        // Test various edge cases based on fuzzer input
        if (offset < Size) {
            uint8_t test_case = Data[offset++] % 8;
            
            switch(test_case) {
                case 0: {
                    // Normal case
                    result.put_(index, values, accumulate);
                    break;
                }
                case 1: {
                    // Test with negative indices
                    index = index - input.numel() / 2;
                    result.put_(index, values, accumulate);
                    break;
                }
                case 2: {
                    // Test with out-of-bounds indices (should wrap around)
                    index = index * 2 + input.numel();
                    result.put_(index, values, accumulate);
                    break;
                }
                case 3: {
                    // Test with empty index
                    torch::Tensor empty_index = torch::empty({0}, torch::kLong);
                    torch::Tensor empty_values = torch::empty({0}, input.dtype());
                    result.put_(empty_index, empty_values, accumulate);
                    break;
                }
                case 4: {
                    // Test with duplicate indices
                    if (index.numel() > 1) {
                        index = torch::cat({index, index.slice(0, 0, 1)});
                        values = torch::cat({values, values.slice(0, 0, 1)});
                    }
                    result.put_(index, values, accumulate);
                    break;
                }
                case 5: {
                    // Test with all same index
                    if (index.numel() > 0) {
                        index.fill_(index[0].item<int64_t>() % input.numel());
                    }
                    result.put_(index, values, accumulate);
                    break;
                }
                case 6: {
                    // Test with very large indices
                    index = index.abs() * 1000000;
                    result.put_(index, values, accumulate);
                    break;
                }
                case 7: {
                    // Test on different device if available
                    if (torch::cuda::is_available() && offset < Size && (Data[offset++] & 0x01)) {
                        result = result.to(torch::kCUDA);
                        index = index.to(torch::kCUDA);
                        values = values.to(torch::kCUDA);
                        result.put_(index, values, accumulate);
                        result = result.to(torch::kCPU);
                    } else {
                        result.put_(index, values, accumulate);
                    }
                    break;
                }
            }
        } else {
            // Default case
            result.put_(index, values, accumulate);
        }
        
        // Additional operations to increase coverage
        if (offset < Size) {
            uint8_t extra_ops = Data[offset++];
            
            // Test chained operations
            if (extra_ops & 0x01) {
                torch::Tensor index2 = torch::randint(0, result.numel(), {2});
                torch::Tensor values2 = torch::ones({2}, result.dtype());
                result.put_(index2, values2, !accumulate);
            }
            
            // Test with different shapes
            if ((extra_ops & 0x02) && result.numel() > 1) {
                result = result.reshape({-1});
                torch::Tensor flat_index = torch::arange(std::min(static_cast<int64_t>(3), result.numel()));
                torch::Tensor flat_values = torch::zeros({flat_index.numel()}, result.dtype());
                result.put_(flat_index, flat_values, accumulate);
            }
            
            // Test with complex dtypes if applicable
            if ((extra_ops & 0x04) && !result.is_complex()) {
                auto complex_result = torch::complex(result, torch::zeros_like(result));
                auto complex_values = torch::complex(values, torch::zeros_like(values));
                complex_result.put_(index.slice(0, 0, std::min(index.numel(), complex_values.numel())), 
                                   complex_values.slice(0, 0, std::min(index.numel(), complex_values.numel())), 
                                   accumulate);
            }
        }
        
        // Validate result
        if (result.numel() != input.numel()) {
            std::cerr << "Warning: Result size changed from " << input.numel() 
                     << " to " << result.numel() << std::endl;
        }
        
        // Check for NaN or Inf in result
        if (result.dtype().isFloatingPoint() || result.dtype().isComplex()) {
            bool has_nan = torch::any(torch::isnan(result)).item<bool>();
            bool has_inf = torch::any(torch::isinf(result)).item<bool>();
            
            if (has_nan || has_inf) {
                // This is acceptable behavior for edge cases
                #ifdef DEBUG_FUZZ
                std::cout << "Result contains NaN or Inf (acceptable for edge cases)" << std::endl;
                #endif
            }
        }
        
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are often expected for invalid operations
        #ifdef DEBUG_FUZZ
        std::cout << "PyTorch error (expected for edge cases): " << e.what() << std::endl;
        #endif
        return 0; // Continue fuzzing
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // Discard input only for unexpected exceptions
    }
    
    return 0;
}