#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte to determine number of tensors
        if (Size < 1) {
            return 0;
        }
        
        // Determine number of tensors to concatenate (1-4)
        uint8_t num_tensors = (Data[offset++] % 4) + 1;
        
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
        
        // Determine dimension to concatenate along
        int64_t dim = 0;
        if (offset < Size) {
            // Get a dimension value from the input data
            // Allow negative dimensions for testing edge cases
            int8_t dim_value;
            std::memcpy(&dim_value, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            
            // If the tensor has dimensions, use the input to select one
            if (!tensors[0].sizes().empty()) {
                // Allow negative indexing (PyTorch supports negative dims)
                dim = dim_value;
            }
        }
        
        // Try concatenating the tensors
        try {
            torch::Tensor result = torch::cat(tensors, dim);
        } catch (const c10::Error& e) {
            // Expected exceptions from PyTorch operations are fine
        }
        
        // Try the concatenate function (alias of cat)
        try {
            torch::Tensor result = torch::concatenate(tensors, dim);
        } catch (const c10::Error& e) {
            // Expected exceptions from PyTorch operations are fine
        }
        
        // Try with different options for dim_size if we have enough data
        if (offset + sizeof(int64_t) <= Size) {
            int64_t new_dim;
            std::memcpy(&new_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            try {
                torch::Tensor result = torch::cat(tensors, new_dim);
            } catch (const c10::Error& e) {
                // Expected exceptions from PyTorch operations are fine
            }
            
            try {
                torch::Tensor result = torch::concatenate(tensors, new_dim);
            } catch (const c10::Error& e) {
                // Expected exceptions from PyTorch operations are fine
            }
        }
        
        // Try with requires_grad option by creating tensors with the option
        if (offset < Size) {
            try {
                bool requires_grad = Data[offset] % 2 == 0;
                std::vector<torch::Tensor> grad_tensors;
                for (const auto& tensor : tensors) {
                    grad_tensors.push_back(tensor.requires_grad_(requires_grad));
                }
                torch::Tensor result = torch::cat(grad_tensors, dim);
            } catch (const c10::Error& e) {
                // Expected exceptions from PyTorch operations are fine
            }
            
            try {
                bool requires_grad = Data[offset] % 2 == 0;
                std::vector<torch::Tensor> grad_tensors;
                for (const auto& tensor : tensors) {
                    grad_tensors.push_back(tensor.requires_grad_(requires_grad));
                }
                torch::Tensor result = torch::concatenate(grad_tensors, dim);
            } catch (const c10::Error& e) {
                // Expected exceptions from PyTorch operations are fine
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
