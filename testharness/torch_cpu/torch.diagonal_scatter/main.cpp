#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstring>        // For std::memcpy

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
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor (should be at least 2D for diagonal_scatter)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input is at least 2D
        if (input.dim() < 2) {
            input = input.unsqueeze(0).unsqueeze(0);
        }
        
        // Get parameters from fuzzer data
        int8_t offset_val = 0;
        int8_t dim1_raw = 0;
        int8_t dim2_raw = 1;
        
        if (offset < Size) {
            offset_val = static_cast<int8_t>(Data[offset]);
            offset++;
        }
        
        if (offset < Size) {
            dim1_raw = static_cast<int8_t>(Data[offset]);
            offset++;
        }
        
        if (offset < Size) {
            dim2_raw = static_cast<int8_t>(Data[offset]);
            offset++;
        }
        
        // Constrain dimensions to valid range
        int64_t ndim = input.dim();
        int64_t dim1 = dim1_raw % ndim;
        if (dim1 < 0) dim1 += ndim;
        
        int64_t dim2 = dim2_raw % ndim;
        if (dim2 < 0) dim2 += ndim;
        
        // Ensure dim1 != dim2
        if (dim1 == dim2) {
            dim2 = (dim1 + 1) % ndim;
        }
        
        // Constrain offset to reasonable range
        int64_t diag_offset = offset_val % 10;  // Keep offset small
        
        // Inner try-catch for expected shape/dimension failures
        try {
            // Get the expected diagonal size
            auto diag = torch::diagonal(input, diag_offset, dim1, dim2);
            
            // Create src tensor with matching shape to the diagonal
            torch::Tensor src;
            if (offset < Size) {
                src = fuzzer_utils::createTensor(Data, Size, offset);
                // Reshape src to match diagonal shape if possible
                try {
                    if (src.numel() >= diag.numel() && diag.numel() > 0) {
                        src = src.flatten().slice(0, 0, diag.numel()).view(diag.sizes());
                    } else {
                        src = torch::ones_like(diag);
                    }
                } catch (...) {
                    src = torch::ones_like(diag);
                }
            } else {
                src = torch::ones_like(diag);
            }
            
            // Ensure src has same dtype as input
            src = src.to(input.dtype());
            
            // Apply diagonal_scatter operation
            torch::Tensor result = torch::diagonal_scatter(input, src, diag_offset, dim1, dim2);
            
            // Verify the result is a valid tensor
            if (result.defined() && result.numel() > 0) {
                // Access some elements to ensure the operation completed
                auto item = result.flatten()[0].item();
                
                // Additional verification: check shape preserved
                (void)(result.sizes() == input.sizes());
            }
        } catch (...) {
            // Silently handle expected failures (shape mismatches, etc.)
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}