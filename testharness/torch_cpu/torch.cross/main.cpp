#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have enough data for the second tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a dimension value for cross product if there's data left
        int64_t dim = -1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Try different variants of cross product
        try {
            // Variant 1: Basic cross product
            torch::Tensor result1 = torch::cross(input1, input2);
        } 
        catch (const std::exception&) {
            // Continue to next variant
        }
        
        try {
            // Variant 2: Cross product with specified dimension
            torch::Tensor result2 = torch::cross(input1, input2, dim);
        }
        catch (const std::exception&) {
            // Continue to next variant
        }
        
        // Try with different input types
        try {
            // Convert to float for numerical stability
            torch::Tensor float_input1 = input1.to(torch::kFloat);
            torch::Tensor float_input2 = input2.to(torch::kFloat);
            
            torch::Tensor result3 = torch::cross(float_input1, float_input2);
        }
        catch (const std::exception&) {
            // Continue
        }
        
        // Try with different shapes
        try {
            // Reshape tensors if possible
            if (input1.numel() >= 3 && input2.numel() >= 3) {
                auto shape1 = input1.sizes().vec();
                auto shape2 = input2.sizes().vec();
                
                // Try to reshape to vectors
                torch::Tensor reshaped1 = input1.reshape({-1, 3});
                torch::Tensor reshaped2 = input2.reshape({-1, 3});
                
                torch::Tensor result4 = torch::cross(reshaped1, reshaped2);
            }
        }
        catch (const std::exception&) {
            // Continue
        }
        
        // Try with tensors that have exactly 3 elements in the last dimension
        try {
            if (input1.dim() > 0 && input2.dim() > 0) {
                std::vector<int64_t> shape1(input1.dim(), 1);
                std::vector<int64_t> shape2(input2.dim(), 1);
                
                // Set last dimension to 3 for cross product
                shape1[shape1.size() - 1] = 3;
                shape2[shape2.size() - 1] = 3;
                
                torch::Tensor shaped1 = torch::ones(shape1, input1.options());
                torch::Tensor shaped2 = torch::ones(shape2, input2.options());
                
                torch::Tensor result5 = torch::cross(shaped1, shaped2);
            }
        }
        catch (const std::exception&) {
            // Continue
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}