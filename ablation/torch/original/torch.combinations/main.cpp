#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstdint>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    // Minimum bytes needed: 1 for params, 2 for tensor metadata (dtype, rank)
    if (size < 3) {
        return 0;
    }

    try
    {
        size_t offset = 0;
        
        // Parse control parameters first
        uint8_t control_byte = data[offset++];
        
        // Extract r value (0-15 range to test various combination lengths)
        int64_t r = static_cast<int64_t>(control_byte & 0x0F);
        
        // Extract with_replacement flag
        bool with_replacement = (control_byte & 0x10) != 0;
        
        // Use remaining bits to decide on special cases
        bool use_empty_tensor = (control_byte & 0x20) != 0;
        bool use_scalar_tensor = (control_byte & 0x40) != 0;
        bool use_large_r = (control_byte & 0x80) != 0;
        
        torch::Tensor input_tensor;
        
        if (use_empty_tensor && offset < size) {
            // Create an empty 1D tensor
            uint8_t dtype_selector = data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            auto options = torch::TensorOptions().dtype(dtype);
            input_tensor = torch::empty({0}, options);
        }
        else if (use_scalar_tensor && offset < size) {
            // Create a scalar (0D) tensor - should fail for combinations
            uint8_t dtype_selector = data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            auto options = torch::TensorOptions().dtype(dtype);
            input_tensor = torch::ones({}, options);
        }
        else {
            // Create a regular 1D tensor from fuzzer input
            // We need to ensure it's 1D for combinations
            if (offset + 2 > size) {
                return 0;
            }
            
            // Parse dtype
            uint8_t dtype_selector = data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Parse length of 1D tensor
            uint8_t length_byte = data[offset++];
            int64_t tensor_length = static_cast<int64_t>(length_byte % 32); // Cap at 32 for performance
            
            // Create tensor data
            size_t dtype_size = c10::elementSize(dtype);
            size_t bytes_needed = tensor_length * dtype_size;
            
            std::vector<uint8_t> tensor_data(bytes_needed, 0);
            size_t bytes_available = (offset < size) ? (size - offset) : 0;
            size_t bytes_to_copy = std::min(bytes_needed, bytes_available);
            
            if (bytes_to_copy > 0) {
                std::memcpy(tensor_data.data(), data + offset, bytes_to_copy);
                offset += bytes_to_copy;
            }
            
            // Create 1D tensor
            auto options = torch::TensorOptions().dtype(dtype);
            if (tensor_length > 0) {
                input_tensor = torch::from_blob(tensor_data.data(), {tensor_length}, options).clone();
            } else {
                input_tensor = torch::empty({0}, options);
            }
        }
        
        // Adjust r for special cases
        if (use_large_r) {
            // Test r larger than input size
            r = input_tensor.numel() + (r % 10) + 1;
        }
        
        // Handle negative r edge case
        if (offset < size && data[offset] & 0x01) {
            r = -r;
        }
        
#ifdef DEBUG_FUZZ
        std::cout << "Input tensor shape: " << input_tensor.sizes() 
                  << ", dtype: " << input_tensor.dtype()
                  << ", r: " << r
                  << ", with_replacement: " << with_replacement << std::endl;
#endif
        
        // Call torch.combinations
        torch::Tensor result;
        
        // combinations expects a 1D tensor, handle non-1D cases
        if (input_tensor.dim() != 1) {
            // Try to flatten or reshape
            if (input_tensor.numel() > 0) {
                input_tensor = input_tensor.flatten();
            }
        }
        
        // Call the actual function
        result = torch::combinations(input_tensor, r, with_replacement);
        
#ifdef DEBUG_FUZZ
        std::cout << "Result shape: " << result.sizes() 
                  << ", dtype: " << result.dtype() << std::endl;
#endif
        
        // Perform some basic validation/operations on result to ensure it's valid
        if (result.numel() > 0) {
            // Check result properties
            auto result_dim = result.dim();
            if (result_dim != 2 && result_dim != 0) {
                std::cerr << "Unexpected result dimension: " << result_dim << std::endl;
            }
            
            // If r > 0 and input has elements, check second dimension matches r
            if (r > 0 && input_tensor.numel() > 0 && result.numel() > 0) {
                if (result.dim() == 2 && result.size(1) != r) {
                    std::cerr << "Result second dimension doesn't match r: " 
                              << result.size(1) << " vs " << r << std::endl;
                }
            }
            
            // Trigger computation
            if (result.is_floating_point() || result.is_complex()) {
                auto sum_val = result.sum();
                (void)sum_val;
            } else {
                auto max_val = result.max();
                (void)max_val;
            }
        }
        
        // Test edge cases with different dtypes
        if (offset + 1 < size) {
            uint8_t extra_test = data[offset++];
            if (extra_test & 0x01) {
                // Test with float tensor
                auto float_input = input_tensor.to(torch::kFloat32);
                auto float_result = torch::combinations(float_input, r, with_replacement);
                (void)float_result;
            }
            if (extra_test & 0x02) {
                // Test with int64 tensor
                auto int_input = input_tensor.to(torch::kInt64);
                auto int_result = torch::combinations(int_input, r, with_replacement);
                (void)int_result;
            }
        }
        
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are expected for invalid inputs
#ifdef DEBUG_FUZZ
        std::cout << "c10::Error caught: " << e.what() << std::endl;
#endif
        return 0; // Keep testing with other inputs
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // Unexpected error, discard input
    }
    catch (...)
    {
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }
    
    return 0;
}