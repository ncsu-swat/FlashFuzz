#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For INFINITY

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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip empty tensors
        if (input.numel() == 0) {
            return 0;
        }
        
        // Extract parameters for native_norm
        if (offset + 2 <= Size) {
            // Extract p value (norm type)
            double p;
            uint8_t p_selector = Data[offset++];
            // Map to common p values: 0, 1, 2, inf, -inf, or fractional
            switch (p_selector % 7) {
                case 0: p = 0.0; break;
                case 1: p = 1.0; break;
                case 2: p = 2.0; break;
                case 3: p = std::numeric_limits<double>::infinity(); break;
                case 4: p = -std::numeric_limits<double>::infinity(); break;
                case 5: p = 0.5 + (p_selector / 7) % 10 / 10.0; break;
                default: p = 2.0; break;
            }
            
            // Extract dim value, bounded to valid range
            int64_t dim = 0;
            if (offset < Size) {
                int64_t ndim = input.dim();
                if (ndim > 0) {
                    dim = static_cast<int64_t>(Data[offset++]) % ndim;
                    // Handle negative dimensions too
                    if (offset < Size && (Data[offset++] & 0x1)) {
                        dim = dim - ndim;
                    }
                } else {
                    offset++;
                }
            }
            
            // Extract keepdim flag
            bool keepdim = false;
            if (offset < Size) {
                keepdim = Data[offset++] & 0x1;
            }
            
            // Extract dtype option
            c10::optional<torch::ScalarType> dtype = c10::nullopt;
            if (offset < Size) {
                uint8_t dtype_selector = Data[offset++];
                if (dtype_selector & 0x1) {
                    // Use float types for norm computation
                    switch ((dtype_selector >> 1) % 4) {
                        case 0: dtype = torch::kFloat32; break;
                        case 1: dtype = torch::kFloat64; break;
                        case 2: dtype = torch::kFloat16; break;
                        default: dtype = c10::nullopt; break;
                    }
                }
            }
            
            // Call native_norm with different parameter combinations
            // Inner try-catch for expected PyTorch errors
            try {
                // Call with all parameters
                torch::Tensor result1 = torch::native_norm(input, c10::optional<torch::Scalar>(p), {dim}, keepdim, dtype);
            } catch (const c10::Error& e) {
                // PyTorch specific errors are expected (shape mismatches, etc.)
            }
            
            try {
                // Call with only p parameter (returns scalar norm)
                torch::Tensor result2 = torch::native_norm(input, p);
            } catch (const c10::Error& e) {
                // PyTorch specific errors are expected
            }
            
            try {
                // Call with default p (2.0)
                torch::Tensor result3 = torch::native_norm(input);
            } catch (const c10::Error& e) {
                // PyTorch specific errors are expected
            }
            
            // Try with multiple dimensions if tensor has enough dims
            if (input.dim() >= 2 && offset < Size) {
                try {
                    int64_t dim2 = (dim + 1) % input.dim();
                    std::vector<int64_t> dims = {dim, dim2};
                    torch::Tensor result4 = torch::native_norm(input, c10::optional<torch::Scalar>(p), dims, keepdim, dtype);
                } catch (const c10::Error& e) {
                    // Expected for invalid dimension combinations
                }
            }
        } else {
            // Not enough data for parameters, try with defaults
            try {
                torch::Tensor result = torch::native_norm(input);
            } catch (const c10::Error& e) {
                // PyTorch specific errors are expected and handled
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}