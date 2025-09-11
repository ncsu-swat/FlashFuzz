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
        
        // Determine number of tensors to create (1-4)
        uint8_t num_tensors = (Size > 0) ? (Data[0] % 4) + 1 : 1;
        offset++;
        
        // Create a vector to hold our tensors
        std::vector<torch::Tensor> tensors;
        
        // Create tensors
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                tensors.push_back(tensor);
            } catch (const std::exception& e) {
                // If we can't create a tensor, just continue with what we have
                break;
            }
        }
        
        // Need at least one tensor to proceed
        if (tensors.empty()) {
            return 0;
        }
        
        // Apply torch.dstack operation
        torch::Tensor result;
        
        // Test different scenarios based on number of tensors
        if (tensors.size() == 1) {
            // Single tensor case - dstack with itself in different ways
            result = torch::dstack({tensors[0]});
        } else {
            // Multiple tensors case
            result = torch::dstack(tensors);
        }
        
        // Optional: Test edge cases with empty tensors if we have enough tensors
        if (tensors.size() >= 2) {
            // Create an empty tensor with same dtype as first tensor
            std::vector<int64_t> empty_shape = {0};
            torch::Tensor empty_tensor = torch::empty(empty_shape, tensors[0].options());
            
            // Try dstacking with an empty tensor
            std::vector<torch::Tensor> tensors_with_empty = tensors;
            tensors_with_empty.push_back(empty_tensor);
            
            torch::Tensor result_with_empty = torch::dstack(tensors_with_empty);
        }
        
        // Test with tensors that have different dtypes if we have multiple tensors
        if (tensors.size() >= 2 && offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto new_dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Convert one tensor to a different dtype
            torch::Tensor converted = tensors[0].to(new_dtype);
            std::vector<torch::Tensor> mixed_tensors = {converted};
            
            // Add the rest of the tensors
            for (size_t i = 1; i < tensors.size(); ++i) {
                mixed_tensors.push_back(tensors[i]);
            }
            
            // Try dstacking tensors with mixed dtypes
            try {
                torch::Tensor mixed_result = torch::dstack(mixed_tensors);
            } catch (const std::exception& e) {
                // Expected to potentially fail with dtype mismatch
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
