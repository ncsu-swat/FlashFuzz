#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create a tensor
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.alias_copy operation
        torch::Tensor result = torch::alias_copy(input_tensor);
        
        // Verify that the result is an alias of the input
        if (!result.is_alias_of(input_tensor)) {
            throw std::runtime_error("Result is not an alias of input tensor");
        }
        
        // Verify that modifying the alias affects the original
        if (result.numel() > 0 && result.is_floating_point()) {
            // For floating point tensors, we can modify values
            torch::Tensor original_copy = input_tensor.clone();
            result.fill_(42.0);
            
            // Check if input_tensor was modified
            if (torch::allclose(input_tensor, original_copy)) {
                throw std::runtime_error("Modifying alias did not affect original tensor");
            }
        }
        
        // Try with different tensor formats if we have more data
        if (Size - offset >= 1) {
            uint8_t option_byte = Data[offset++];
            
            // Test with different tensor formats by creating new tensors
            torch::Tensor format_tensor;
            if (option_byte % 3 == 1 && input_tensor.dim() >= 4) {
                // Try to create channels_last format tensor
                format_tensor = input_tensor.to(torch::MemoryFormat::ChannelsLast);
            } else if (option_byte % 3 == 2) {
                // Create contiguous tensor
                format_tensor = input_tensor.contiguous();
            } else {
                format_tensor = input_tensor;
            }
            
            // Apply alias_copy to the formatted tensor
            torch::Tensor result_with_format = torch::alias_copy(format_tensor);
            
            // Verify it's still an alias
            if (!result_with_format.is_alias_of(format_tensor)) {
                throw std::runtime_error("Result with format is not an alias of input tensor");
            }
        }
        
        // Test edge cases with special tensors if we have more data
        if (Size - offset >= 1) {
            uint8_t edge_case = Data[offset++];
            
            // Create some edge case tensors
            torch::Tensor zero_dim_tensor = torch::tensor(5.0);
            torch::Tensor empty_tensor = torch::empty({0});
            torch::Tensor scalar_tensor = torch::scalar_tensor(3.14);
            
            // Select which edge case to test based on fuzzer data
            torch::Tensor edge_input;
            switch (edge_case % 3) {
                case 0:
                    edge_input = zero_dim_tensor;
                    break;
                case 1:
                    edge_input = empty_tensor;
                    break;
                case 2:
                    edge_input = scalar_tensor;
                    break;
            }
            
            // Apply alias_copy to the edge case
            torch::Tensor edge_result = torch::alias_copy(edge_input);
            
            // Verify it's an alias
            if (!edge_result.is_alias_of(edge_input)) {
                throw std::runtime_error("Edge case result is not an alias of input tensor");
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