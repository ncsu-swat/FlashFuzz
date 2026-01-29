#include "fuzzer_utils.h"
#include <iostream>
#include <ATen/autocast_mode.h>

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
        
        // Need minimum data to work with
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to ensure PyTorch is initialized
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test autocast functionality based on fuzzer input
        bool enabled = static_cast<bool>(Data[offset % Size] & 0x01);
        offset++;
        
        // Set up autocast context on CPU
        try {
            at::autocast::set_autocast_enabled(at::kCPU, enabled);
            
            if (enabled) {
                // Create some tensors with autocast enabled
                torch::Tensor t1 = fuzzer_utils::createTensor(Data, Size, offset);
                torch::Tensor t2 = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Perform operations that might populate the autocast cache
                // Use inner try-catch for expected shape mismatch errors
                try {
                    torch::Tensor result = t1 + t2;
                    result = torch::sum(result);
                } catch (...) {
                    // Shape mismatches are expected, ignore silently
                }
                
                try {
                    // Matrix operations may use cached weight conversions
                    if (t1.dim() >= 2 && t2.dim() >= 2) {
                        auto t1_2d = t1.view({t1.size(0), -1});
                        auto t2_2d = t2.view({t2.size(0), -1});
                        if (t1_2d.size(1) == t2_2d.size(0)) {
                            torch::Tensor mm_result = torch::mm(t1_2d, t2_2d);
                        }
                    }
                } catch (...) {
                    // Expected failures for incompatible shapes
                }
            }
            
            // Test the clear_autocast_cache function - this is the main API under test
            at::autocast::clear_cache();
            
            // Test with different dtype settings
            uint8_t dtype_selector = Data[offset % Size];
            offset++;
            
            try {
                if (dtype_selector % 2 == 0) {
                    at::autocast::set_autocast_dtype(at::kCPU, at::kBFloat16);
                } else {
                    at::autocast::set_autocast_dtype(at::kCPU, at::kHalf);
                }
            } catch (...) {
                // dtype may not be supported on this platform
            }
            
            // Perform more operations with potentially different dtype
            if (enabled) {
                torch::Tensor t3 = fuzzer_utils::createTensor(Data, Size, offset);
                
                try {
                    torch::Tensor result = torch::sin(t3);
                    result = torch::cos(result);
                    result = torch::exp(result);
                } catch (...) {
                    // Numeric operations may fail for certain inputs
                }
            }
            
            // Clear cache again after operations
            at::autocast::clear_cache();
            
            // Test increment/decrement nesting
            at::autocast::increment_nesting();
            at::autocast::clear_cache();
            at::autocast::decrement_nesting();
            
            // Final clear
            at::autocast::clear_cache();
            
            // Reset autocast state
            at::autocast::set_autocast_enabled(at::kCPU, false);
            
        } catch (...) {
            // Autocast operations may not be available, reset and continue
            try {
                at::autocast::set_autocast_enabled(at::kCPU, false);
            } catch (...) {}
        }
        
        // Always call clear_cache at the end to ensure the API is exercised
        at::autocast::clear_cache();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}