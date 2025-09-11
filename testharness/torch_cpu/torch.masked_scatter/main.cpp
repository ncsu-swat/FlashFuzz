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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create the input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create the mask tensor (should be a boolean tensor)
        torch::Tensor mask;
        if (offset < Size) {
            mask = fuzzer_utils::createTensor(Data, Size, offset);
            // Convert to boolean tensor if not already
            mask = mask.to(torch::kBool);
        } else {
            // If we don't have enough data, create a mask of the same shape as input
            mask = torch::ones_like(input_tensor, torch::kBool);
        }
        
        // Create the source tensor
        torch::Tensor source;
        if (offset < Size) {
            source = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a source tensor with some values
            source = torch::ones_like(input_tensor);
        }
        
        // Apply masked_scatter operation
        torch::Tensor result = input_tensor.masked_scatter(mask, source);
        
        // Try different variants of the operation
        if (Size > offset + 1) {
            // Create a smaller mask to test broadcasting
            std::vector<int64_t> smaller_shape;
            for (size_t i = 0; i < input_tensor.dim(); i++) {
                if (i < input_tensor.dim() - 1) {
                    smaller_shape.push_back(input_tensor.size(i));
                } else {
                    smaller_shape.push_back(1);
                }
            }
            
            if (!smaller_shape.empty()) {
                torch::Tensor smaller_mask = torch::ones(smaller_shape, torch::kBool);
                torch::Tensor result2 = input_tensor.masked_scatter(smaller_mask, source);
            }
            
            // Try with a scalar mask
            bool scalar_mask_value = (Data[offset] % 2 == 0);
            torch::Tensor scalar_mask = torch::tensor(scalar_mask_value, torch::kBool);
            torch::Tensor result3 = input_tensor.masked_scatter(scalar_mask, source);
        }
        
        // Try with different source tensor shapes
        if (Size > offset + 2) {
            // Create a source tensor with different shape
            torch::Tensor different_source = torch::ones({1});
            torch::Tensor result4 = input_tensor.masked_scatter(mask, different_source);
        }
        
        // Try with empty tensors
        if (input_tensor.numel() > 0) {
            torch::Tensor empty_source = torch::ones({0});
            try {
                torch::Tensor result5 = input_tensor.masked_scatter(mask, empty_source);
            } catch (...) {
                // Expected to fail in some cases
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
