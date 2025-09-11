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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract dimension parameter for sum if there's data left
        int64_t dim = 0;
        bool keepdim = false;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Extract keepdim parameter if there's data left
        if (offset < Size) {
            keepdim = Data[offset] & 0x1;
        }
        
        // Apply sum operation with different parameter combinations
        torch::Tensor result;
        
        // Test different variants of sum
        if (input_tensor.dim() > 0) {
            // Test with specific dimension
            result = torch::sum(input_tensor, dim, keepdim);
            
            // Test with named tensors if available
            try {
                if (input_tensor.has_names()) {
                    auto names = input_tensor.names();
                    if (!names.empty() && names[0].has_value()) {
                        auto result_named = torch::sum(input_tensor, names[0].value(), keepdim);
                    }
                }
            } catch (...) {
                // Ignore errors with named tensors
            }
        }
        
        // Test without dimension parameter (sum over all dimensions)
        result = torch::sum(input_tensor);
        
        // Test with empty tensor
        if (input_tensor.numel() > 0) {
            auto empty_tensor = torch::empty({0}, input_tensor.options());
            try {
                auto empty_result = torch::sum(empty_tensor);
            } catch (...) {
                // Ignore expected errors with empty tensors
            }
        }
        
        // Test with different dtypes
        if (input_tensor.dtype() != torch::kBool && 
            input_tensor.dtype() != torch::kBFloat16 && 
            input_tensor.dtype() != torch::kHalf) {
            try {
                auto bool_tensor = input_tensor.to(torch::kBool);
                auto bool_result = torch::sum(bool_tensor);
            } catch (...) {
                // Ignore expected errors
            }
        }
        
        // Test with out-of-bounds dimension
        if (input_tensor.dim() > 0) {
            try {
                int64_t out_of_bounds_dim = input_tensor.dim() + 5;
                auto invalid_result = torch::sum(input_tensor, out_of_bounds_dim, keepdim);
            } catch (...) {
                // Ignore expected errors
            }
            
            try {
                int64_t negative_dim = -input_tensor.dim() - 5;
                auto negative_result = torch::sum(input_tensor, negative_dim, keepdim);
            } catch (...) {
                // Ignore expected errors
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
