#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // view_as_complex_copy requires a tensor with a last dimension of size 2
        // We'll try to use it regardless to test error handling
        try {
            // Apply view_as_complex_copy operation
            torch::Tensor complex_tensor = torch::view_as_complex_copy(input_tensor);
            
            // Verify the result is complex
            if (complex_tensor.defined()) {
                bool is_complex = complex_tensor.is_complex();
                
                // Try to convert back to real (should work if the operation succeeded)
                if (is_complex) {
                    torch::Tensor real_tensor = torch::view_as_real(complex_tensor);
                }
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for invalid inputs
            // We don't need to do anything special here
        }
        
        // Try with a reshaped tensor if possible
        if (input_tensor.defined() && input_tensor.numel() > 1) {
            try {
                // Try to reshape the tensor to have last dimension of size 2
                std::vector<int64_t> new_shape;
                int64_t total_elements = input_tensor.numel();
                
                if (total_elements % 2 == 0) {
                    // If even number of elements, we can reshape to have last dim = 2
                    if (total_elements == 2) {
                        new_shape = {1, 2};
                    } else {
                        new_shape = {total_elements / 2, 2};
                    }
                    
                    torch::Tensor reshaped = input_tensor.reshape(new_shape);
                    torch::Tensor complex_tensor = torch::view_as_complex_copy(reshaped);
                    
                    // Try some operations on the complex tensor
                    if (complex_tensor.defined()) {
                        torch::Tensor abs_val = torch::abs(complex_tensor);
                        torch::Tensor real_part = torch::real(complex_tensor);
                        torch::Tensor imag_part = torch::imag(complex_tensor);
                    }
                }
            } catch (const c10::Error& e) {
                // Expected for invalid operations
            }
        }
        
        // Try with a contiguous tensor if the original isn't
        if (input_tensor.defined() && !input_tensor.is_contiguous()) {
            try {
                torch::Tensor contiguous_tensor = input_tensor.contiguous();
                
                // Try to reshape to have last dimension of size 2 if possible
                int64_t total_elements = contiguous_tensor.numel();
                if (total_elements % 2 == 0 && total_elements > 0) {
                    std::vector<int64_t> new_shape;
                    if (total_elements == 2) {
                        new_shape = {1, 2};
                    } else {
                        new_shape = {total_elements / 2, 2};
                    }
                    
                    torch::Tensor reshaped = contiguous_tensor.reshape(new_shape);
                    torch::Tensor complex_tensor = torch::view_as_complex_copy(reshaped);
                }
            } catch (const c10::Error& e) {
                // Expected for invalid operations
            }
        }
        
        // Try with a tensor of specific shape for complex conversion
        if (offset + 2 < Size) {
            try {
                // Create a tensor specifically for complex conversion
                std::vector<int64_t> shape;
                
                // Get a random number from the data to determine shape complexity
                uint8_t shape_complexity = Data[offset++] % 3;  // 0, 1, or 2
                
                if (shape_complexity == 0) {
                    // Just a single complex number
                    shape = {2};
                } else if (shape_complexity == 1) {
                    // Vector of complex numbers
                    uint8_t length = (Data[offset++] % 8) + 1;  // 1-8 complex numbers
                    shape = {length, 2};
                } else {
                    // Matrix of complex numbers
                    uint8_t rows = (Data[offset++] % 4) + 1;    // 1-4 rows
                    uint8_t cols = (Data[offset++] % 4) + 1;    // 1-4 columns
                    shape = {rows, cols, 2};
                }
                
                // Create a tensor with the right shape for complex conversion
                torch::Tensor shaped_tensor = torch::ones(shape);
                torch::Tensor complex_tensor = torch::view_as_complex_copy(shaped_tensor);
                
                // Try some operations on the complex tensor
                if (complex_tensor.defined()) {
                    torch::Tensor conj = torch::conj(complex_tensor);
                    torch::Tensor angle = torch::angle(complex_tensor);
                }
            } catch (const c10::Error& e) {
                // Expected for invalid operations
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