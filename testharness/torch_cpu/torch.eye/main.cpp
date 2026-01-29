#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Need at least 2 bytes for n and m parameters
        if (Size < 2) {
            return 0;
        }
        
        // Extract n (number of rows) - limit to reasonable size to avoid OOM
        int64_t n = static_cast<int64_t>(Data[offset++]) % 1024;
        
        // Extract m (number of columns) - limit to reasonable size
        int64_t m = static_cast<int64_t>(Data[offset++]) % 1024;
        
        // Determine which variant to use based on fuzzer data
        uint8_t variant = 0;
        if (offset < Size) {
            variant = Data[offset++] % 4;
        }
        
        // Extract dtype (optional)
        torch::ScalarType dtype = torch::kFloat;
        if (offset < Size) {
            dtype = fuzzer_utils::parseDataType(Data[offset++]);
        }
        
        // Create identity matrix with different parameter combinations
        torch::Tensor result;
        
        try {
            // Test different combinations of parameters
            switch (variant) {
                case 0:
                    // eye(n)
                    result = torch::eye(n);
                    break;
                case 1:
                    // eye(n, m)
                    result = torch::eye(n, m);
                    break;
                case 2:
                    // eye(n, options)
                    result = torch::eye(n, torch::TensorOptions().dtype(dtype).device(torch::kCPU));
                    break;
                case 3:
                default:
                    // eye(n, m, options)
                    result = torch::eye(n, m, torch::TensorOptions().dtype(dtype).device(torch::kCPU));
                    break;
            }
        } catch (const c10::Error &e) {
            // Expected errors for invalid parameters (negative dimensions, etc.)
            return 0;
        }
        
        // Perform some operations on the result to ensure it's used
        if (result.defined() && result.numel() > 0) {
            // Test sum operation
            auto sum = result.sum();
            
            // Test trace operation (only valid for 2D tensors)
            if (result.dim() == 2) {
                auto trace = result.trace();
                
                // Test diagonal extraction
                auto diag = result.diag();
                
                // Test transpose
                auto transposed = result.t();
                
                // Verify identity property: diagonal elements should be 1
                auto diag_sum = result.diag().sum();
            }
            
            // Test clone and contiguous
            auto cloned = result.clone();
            auto contig = result.contiguous();
            
            // Test slicing if dimensions allow
            if (result.size(0) > 1 && result.size(1) > 1) {
                auto slice = result.slice(0, 0, result.size(0) / 2);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}