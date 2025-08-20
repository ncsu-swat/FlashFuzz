#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create vec1 and vec2 tensors for addr operation
        torch::Tensor vec1, vec2;
        
        if (offset < Size) {
            vec1 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create a default vector if we don't have enough data
            if (input.dim() > 0) {
                vec1 = torch::ones({input.size(0)});
            } else {
                vec1 = torch::ones({1});
            }
        }
        
        if (offset < Size) {
            vec2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create a default vector if we don't have enough data
            if (input.dim() > 1) {
                vec2 = torch::ones({input.size(1)});
            } else {
                vec2 = torch::ones({1});
            }
        }
        
        // Get alpha and beta values from the input data if available
        double alpha = 1.0;
        double beta = 1.0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&beta, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Try different variants of addr
        try {
            // Basic addr operation
            torch::Tensor result1 = torch::addr(input, vec1, vec2);
            
            // addr with alpha and beta
            torch::Tensor result2 = torch::addr(input, vec1, vec2, alpha, beta);
            
            // addr with out tensor
            torch::Tensor out = torch::zeros_like(input);
            torch::addr_out(out, input, vec1, vec2);
            
            // addr with alpha, beta and out tensor
            torch::Tensor out2 = torch::zeros_like(input);
            torch::addr_out(out2, input, vec1, vec2, alpha, beta);
        } catch (const c10::Error &e) {
            // PyTorch specific errors are expected for invalid inputs
            return 0;
        }
        
        // Try functional variant
        try {
            auto result = at::addr(input, vec1, vec2, alpha, beta);
        } catch (const c10::Error &e) {
            // PyTorch specific errors are expected for invalid inputs
            return 0;
        }
        
        // Try in-place variant
        try {
            torch::Tensor input_copy = input.clone();
            input_copy.addr_(vec1, vec2);
        } catch (const c10::Error &e) {
            // PyTorch specific errors are expected for invalid inputs
            return 0;
        }
        
        // Try in-place variant with alpha and beta
        try {
            torch::Tensor input_copy = input.clone();
            input_copy.addr_(vec1, vec2, alpha, beta);
        } catch (const c10::Error &e) {
            // PyTorch specific errors are expected for invalid inputs
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}