#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 3 tensors for addmm: input, mat1, mat2
        if (Size < 6) // Minimum bytes needed for basic tensor metadata
            return 0;
            
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create mat1 tensor
        torch::Tensor mat1;
        if (offset < Size) {
            mat1 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            mat1 = torch::ones({1, 1});
        }
        
        // Create mat2 tensor
        torch::Tensor mat2;
        if (offset < Size) {
            mat2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            mat2 = torch::ones({1, 1});
        }
        
        // Get scalar values for alpha and beta if there's data left
        double beta = 1.0;
        double alpha = 1.0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&beta, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Apply addmm operation
        // addmm(input, mat1, mat2, beta, alpha) = beta * input + alpha * (mat1 @ mat2)
        torch::Tensor result;
        
        // Try different variants of the operation
        try {
            // Variant 1: Using torch::addmm directly
            result = torch::addmm(input, mat1, mat2, beta, alpha);
        } catch (const std::exception&) {
            // If direct call fails, try alternative approaches
            try {
                // Variant 2: Using tensor method
                result = input.addmm(mat1, mat2, beta, alpha);
            } catch (const std::exception&) {
                // Variant 3: Manually compute the equivalent operation
                try {
                    result = beta * input + alpha * torch::matmul(mat1, mat2);
                } catch (const std::exception&) {
                    // All variants failed, but that's expected for some inputs
                }
            }
        }
        
        // Try to access result to ensure computation is not optimized away
        if (result.defined()) {
            volatile float dummy = 0.0;
            if (result.numel() > 0 && (result.dtype() == torch::kFloat || result.dtype() == torch::kDouble)) {
                dummy = result.item<float>();
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