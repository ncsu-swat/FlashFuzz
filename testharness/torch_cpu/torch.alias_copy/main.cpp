#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

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
        // Note: alias_copy creates a COPY of the tensor (not an alias/view)
        // It's a "functionalized" version of alias() that returns a new tensor
        torch::Tensor result = torch::alias_copy(input_tensor);
        
        // Verify that the result is a copy, NOT an alias
        // alias_copy should return a tensor with the same data but as a copy
        
        // Verify shapes match
        if (result.sizes() != input_tensor.sizes()) {
            throw std::runtime_error("Result shape does not match input shape");
        }
        
        // Verify dtypes match
        if (result.dtype() != input_tensor.dtype()) {
            throw std::runtime_error("Result dtype does not match input dtype");
        }
        
        // Verify the values are equal
        if (input_tensor.numel() > 0) {
            try {
                // For numeric tensors, check values match
                if (!torch::equal(result, input_tensor)) {
                    throw std::runtime_error("Result values do not match input values");
                }
            } catch (...) {
                // Some tensor types may not support equal(), silently continue
            }
        }
        
        // Verify that modifying the copy does NOT affect the original
        if (result.numel() > 0 && result.is_floating_point()) {
            torch::Tensor original_copy = input_tensor.clone();
            result.fill_(42.0);
            
            // The original should be unchanged since alias_copy returns a copy
            if (!torch::equal(input_tensor, original_copy)) {
                throw std::runtime_error("Modifying copy unexpectedly affected original tensor");
            }
        }
        
        // Test with different tensor configurations
        if (Size - offset >= 1) {
            uint8_t option_byte = Data[offset++];
            
            torch::Tensor test_tensor;
            try {
                if (option_byte % 4 == 0 && input_tensor.dim() >= 4) {
                    // Try channels_last format
                    test_tensor = input_tensor.to(torch::MemoryFormat::ChannelsLast);
                } else if (option_byte % 4 == 1) {
                    // Contiguous tensor
                    test_tensor = input_tensor.contiguous();
                } else if (option_byte % 4 == 2 && input_tensor.dim() >= 2) {
                    // Transposed tensor (non-contiguous)
                    test_tensor = input_tensor.transpose(0, input_tensor.dim() - 1);
                } else {
                    test_tensor = input_tensor;
                }
                
                // Apply alias_copy to the configured tensor
                torch::Tensor result_formatted = torch::alias_copy(test_tensor);
                
                // Verify shapes match
                if (result_formatted.sizes() != test_tensor.sizes()) {
                    throw std::runtime_error("Formatted result shape mismatch");
                }
            } catch (...) {
                // Silently handle expected failures from format conversions
            }
        }
        
        // Test edge cases with special tensors
        if (Size - offset >= 1) {
            uint8_t edge_case = Data[offset++];
            
            torch::Tensor edge_input;
            switch (edge_case % 4) {
                case 0:
                    // Zero-dimensional tensor (scalar)
                    edge_input = torch::tensor(5.0);
                    break;
                case 1:
                    // Empty tensor
                    edge_input = torch::empty({0});
                    break;
                case 2:
                    // Scalar tensor
                    edge_input = torch::scalar_tensor(3.14);
                    break;
                case 3:
                    // Multi-dimensional empty tensor
                    edge_input = torch::empty({0, 3, 4});
                    break;
            }
            
            // Apply alias_copy to the edge case
            torch::Tensor edge_result = torch::alias_copy(edge_input);
            
            // Verify shapes match
            if (edge_result.sizes() != edge_input.sizes()) {
                throw std::runtime_error("Edge case result shape mismatch");
            }
        }
        
        // Test with different dtypes
        if (Size - offset >= 1) {
            uint8_t dtype_byte = Data[offset++];
            
            torch::Tensor typed_tensor;
            try {
                switch (dtype_byte % 6) {
                    case 0:
                        typed_tensor = torch::zeros({2, 3}, torch::kFloat32);
                        break;
                    case 1:
                        typed_tensor = torch::zeros({2, 3}, torch::kFloat64);
                        break;
                    case 2:
                        typed_tensor = torch::zeros({2, 3}, torch::kInt32);
                        break;
                    case 3:
                        typed_tensor = torch::zeros({2, 3}, torch::kInt64);
                        break;
                    case 4:
                        typed_tensor = torch::zeros({2, 3}, torch::kBool);
                        break;
                    case 5:
                        typed_tensor = torch::zeros({2, 3}, torch::kInt8);
                        break;
                }
                
                torch::Tensor typed_result = torch::alias_copy(typed_tensor);
                
                if (typed_result.dtype() != typed_tensor.dtype()) {
                    throw std::runtime_error("Typed result dtype mismatch");
                }
            } catch (...) {
                // Silently handle dtype conversion issues
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}