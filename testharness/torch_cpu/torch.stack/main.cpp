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
        
        // Need at least 1 byte to determine number of tensors
        if (Size < 1) {
            return 0;
        }
        
        // Determine number of tensors to stack (1-10)
        uint8_t num_tensors = (Data[offset++] % 10) + 1;
        
        // Create a vector to hold our tensors
        std::vector<torch::Tensor> tensors;
        
        // Create tensors with varying properties
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                tensors.push_back(tensor);
            } catch (const std::exception& e) {
                // If we can't create a tensor, just continue with what we have
                break;
            }
        }
        
        // Need at least one tensor to stack
        if (tensors.empty()) {
            return 0;
        }
        
        // Get dimension to stack along
        int64_t dim = 0;
        if (offset < Size) {
            // Allow negative dimensions for edge case testing
            int8_t dim_value;
            std::memcpy(&dim_value, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            dim = dim_value;
        }
        
        // Apply torch.stack operation
        try {
            torch::Tensor result = torch::stack(tensors, dim);
        } catch (const c10::Error& e) {
            // PyTorch specific exceptions are expected and not a bug in our fuzzer
            return 0;
        }
        
        // Try stacking with out tensor using stack_out
        if (!tensors.empty()) {
            try {
                // Create an output tensor
                torch::Tensor out_tensor = torch::empty({0});
                
                // Try to stack with out tensor using stack_out
                torch::Tensor result = torch::stack_out(out_tensor, tensors, dim);
            } catch (const c10::Error& e) {
                // PyTorch specific exceptions are expected
                return 0;
            }
        }
        
        // Try stacking with different dimensions
        if (offset < Size && !tensors.empty()) {
            try {
                int8_t alt_dim;
                std::memcpy(&alt_dim, Data + offset, sizeof(int8_t));
                torch::Tensor result = torch::stack(tensors, alt_dim);
            } catch (const c10::Error& e) {
                // PyTorch specific exceptions are expected
                return 0;
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
