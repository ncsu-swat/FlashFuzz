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
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test various tensor operations that can be used for testing
        
        // Test tensor equality using allclose
        try {
            torch::Tensor clone_tensor = input_tensor.clone();
            torch::allclose(input_tensor, clone_tensor);
        } catch (const c10::Error &) {
            // Expected to succeed, but might fail for NaN/Inf values
        }
        
        // Test allclose with default tolerances
        try {
            torch::Tensor clone_tensor = input_tensor.clone();
            torch::allclose(input_tensor, clone_tensor);
        } catch (const c10::Error &) {
            // Expected to succeed, but might fail for NaN/Inf values
        }
        
        // Test allclose with different rtol/atol
        if (offset + 8 <= Size) {
            double rtol = 0.0, atol = 0.0;
            std::memcpy(&rtol, Data + offset, sizeof(double));
            offset += sizeof(double);
            std::memcpy(&atol, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Make rtol and atol non-negative
            rtol = std::abs(rtol);
            atol = std::abs(atol);
            
            try {
                torch::Tensor slightly_modified = input_tensor.clone();
                // Add small noise if tensor is not empty
                if (input_tensor.numel() > 0) {
                    slightly_modified += torch::rand_like(input_tensor) * atol * 0.5;
                }
                torch::allclose(input_tensor, slightly_modified, rtol, atol);
            } catch (const c10::Error &) {
                // May fail depending on the values
            }
        }
        
        // Test tensor equality using equal
        try {
            torch::Tensor clone_tensor = input_tensor.clone();
            torch::equal(input_tensor, clone_tensor);
        } catch (const c10::Error &) {
            // Expected to succeed, but might fail for NaN values
        }
        
        // Test making non-contiguous tensor
        try {
            if (input_tensor.dim() > 0 && input_tensor.numel() > 0) {
                torch::Tensor non_contiguous = input_tensor.transpose(0, -1);
                torch::allclose(input_tensor, non_contiguous.contiguous());
            }
        } catch (const c10::Error &) {
            // May fail for certain tensor shapes/types
        }
        
        // Test tensor equality
        try {
            torch::Tensor clone_tensor = input_tensor.clone();
            torch::equal(input_tensor, clone_tensor);
        } catch (const c10::Error &) {
            // Expected to succeed, but might fail for NaN values
        }
        
        // Test creating tensors with various options
        if (offset + 1 <= Size) {
            uint8_t dtype_selector = Data[offset++];
            torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            try {
                // Create a tensor with random values
                torch::Tensor random_tensor = torch::rand({2, 3}, torch::TensorOptions().dtype(dtype));
                
                // Test with specific shape from input tensor if possible
                if (input_tensor.dim() > 0) {
                    torch::Tensor shaped_tensor = torch::rand(input_tensor.sizes(), torch::TensorOptions().dtype(dtype));
                }
            } catch (const c10::Error &) {
                // May fail for certain dtypes
            }
        }
        
        // Test rand_like
        try {
            torch::Tensor rand_tensor = torch::rand_like(input_tensor);
        } catch (const c10::Error &) {
            // May fail for certain tensor types
        }
        
        // Test randn_like
        try {
            torch::Tensor randn_tensor = torch::randn_like(input_tensor);
        } catch (const c10::Error &) {
            // May fail for certain tensor types
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
