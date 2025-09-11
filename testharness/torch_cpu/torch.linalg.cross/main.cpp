#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least 2 tensors for cross product
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have enough data for the second tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get dim parameter (optional)
        int64_t dim = -1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Try different variants of the cross product operation
        try {
            // Variant 1: Default dim
            torch::Tensor result1 = torch::cross(input1, input2);
        } catch (...) {
            // Ignore exceptions from this variant
        }
        
        try {
            // Variant 2: With explicit dim
            torch::Tensor result2 = torch::cross(input1, input2, dim);
        } catch (...) {
            // Ignore exceptions from this variant
        }
        
        // Try with named arguments
        try {
            torch::Tensor result3 = torch::cross(input1, input2, c10::nullopt);
        } catch (...) {
            // Ignore exceptions from this variant
        }
        
        // Try with different data types
        try {
            auto input1_float = input1.to(torch::kFloat);
            auto input2_float = input2.to(torch::kFloat);
            torch::Tensor result4 = torch::cross(input1_float, input2_float);
        } catch (...) {
            // Ignore exceptions from this variant
        }
        
        // Try with different shapes
        try {
            // Reshape tensors if possible
            if (input1.numel() >= 3 && input2.numel() >= 3) {
                auto reshaped1 = input1.reshape({-1, 3});
                auto reshaped2 = input2.reshape({-1, 3});
                torch::Tensor result5 = torch::cross(reshaped1, reshaped2);
            }
        } catch (...) {
            // Ignore exceptions from this variant
        }
        
        // Try with broadcasting
        try {
            if (input1.dim() > 0 && input2.dim() > 0) {
                // Attempt to create broadcastable tensors
                auto shape1 = input1.sizes().vec();
                auto shape2 = input2.sizes().vec();
                
                if (!shape1.empty() && !shape2.empty()) {
                    shape1[0] = 1;
                    auto broadcasted1 = input1.expand(shape2);
                    auto broadcasted2 = input2.expand(shape1);
                    torch::Tensor result6 = torch::cross(broadcasted1, broadcasted2);
                }
            }
        } catch (...) {
            // Ignore exceptions from this variant
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
