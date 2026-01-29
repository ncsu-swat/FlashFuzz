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
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // view_as_complex_copy requires a tensor with a last dimension of size 2
        // and the tensor must be float or double type
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
        } catch (const std::exception& e) {
            // Other standard exceptions from invalid operations
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
                    
                    // Convert to float if not already (view_as_complex requires float/double)
                    torch::Tensor float_tensor = input_tensor.to(torch::kFloat32);
                    torch::Tensor reshaped = float_tensor.reshape(new_shape);
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
            } catch (const std::exception& e) {
                // Other standard exceptions
            }
        }
        
        // Try with different dtypes - view_as_complex_copy supports float, double, half
        if (offset + 2 < Size) {
            try {
                uint8_t dtype_selector = Data[offset++] % 3;
                torch::ScalarType dtype;
                
                switch (dtype_selector) {
                    case 0: dtype = torch::kFloat32; break;
                    case 1: dtype = torch::kFloat64; break;
                    case 2: dtype = torch::kFloat16; break;
                    default: dtype = torch::kFloat32; break;
                }
                
                // Get shape parameters
                uint8_t shape_complexity = Data[offset++] % 3;
                std::vector<int64_t> shape;
                
                if (shape_complexity == 0) {
                    // Just a single complex number
                    shape = {2};
                } else if (shape_complexity == 1) {
                    // Vector of complex numbers
                    uint8_t length = (offset < Size) ? (Data[offset++] % 8) + 1 : 2;
                    shape = {length, 2};
                } else {
                    // Matrix of complex numbers
                    uint8_t rows = (offset < Size) ? (Data[offset++] % 4) + 1 : 2;
                    uint8_t cols = (offset < Size) ? (Data[offset++] % 4) + 1 : 2;
                    shape = {rows, cols, 2};
                }
                
                // Create a tensor with the right shape and dtype for complex conversion
                torch::Tensor shaped_tensor = torch::randn(shape, torch::dtype(dtype));
                torch::Tensor complex_tensor = torch::view_as_complex_copy(shaped_tensor);
                
                // Try some operations on the complex tensor
                if (complex_tensor.defined()) {
                    torch::Tensor conj = torch::conj(complex_tensor);
                    torch::Tensor angle = torch::angle(complex_tensor);
                    torch::Tensor abs_val = torch::abs(complex_tensor);
                    
                    // Verify round-trip conversion
                    torch::Tensor back_to_real = torch::view_as_real_copy(complex_tensor);
                }
            } catch (const c10::Error& e) {
                // Expected for invalid operations
            } catch (const std::exception& e) {
                // Other standard exceptions
            }
        }
        
        // Test with contiguous tensor explicitly
        if (offset + 1 < Size) {
            try {
                uint8_t size_param = (Data[offset++] % 10) + 1;
                
                // Create a contiguous tensor with proper shape
                torch::Tensor base_tensor = torch::randn({size_param, 2}, torch::kFloat32);
                
                // Ensure it's contiguous
                torch::Tensor contiguous_tensor = base_tensor.contiguous();
                
                torch::Tensor complex_tensor = torch::view_as_complex_copy(contiguous_tensor);
                
                if (complex_tensor.defined()) {
                    // Test arithmetic operations on complex result
                    torch::Tensor doubled = complex_tensor * 2.0;
                    torch::Tensor added = complex_tensor + complex_tensor;
                }
            } catch (const c10::Error& e) {
                // Expected for invalid operations
            } catch (const std::exception& e) {
                // Other standard exceptions
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}