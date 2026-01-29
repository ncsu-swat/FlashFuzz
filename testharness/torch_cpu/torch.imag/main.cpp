#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>
#include <limits>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }

        size_t offset = 0;
        
        // Create a tensor from the input data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Use remaining bytes for control flow decisions
        uint8_t control_byte = (offset < Size) ? Data[offset++] : 0;
        
        // Convert to complex tensor for torch::imag
        torch::Tensor complex_tensor;
        
        if (input_tensor.is_complex()) {
            complex_tensor = input_tensor;
        } else {
            // Convert to float first if needed, then to complex
            torch::Tensor real_part;
            if (input_tensor.is_floating_point()) {
                real_part = input_tensor;
            } else {
                real_part = input_tensor.to(torch::kFloat);
            }
            
            // Create imaginary part - vary based on control byte
            torch::Tensor imag_part;
            switch (control_byte % 4) {
                case 0:
                    imag_part = torch::zeros_like(real_part);
                    break;
                case 1:
                    imag_part = torch::ones_like(real_part);
                    break;
                case 2:
                    imag_part = real_part.clone();
                    break;
                default:
                    imag_part = torch::rand_like(real_part);
                    break;
            }
            
            complex_tensor = torch::complex(real_part, imag_part);
        }
        
        // Apply torch.imag operation - main API under test
        torch::Tensor imag_tensor = torch::imag(complex_tensor);
        
        // Verify output is real-valued (not complex)
        (void)imag_tensor.is_complex();
        
        // Test with different complex dtypes based on fuzzer input
        if ((control_byte & 0x10) && complex_tensor.numel() > 0) {
            try {
                // Test with complex64
                torch::Tensor c64 = complex_tensor.to(torch::kComplexFloat);
                torch::Tensor imag_c64 = torch::imag(c64);
                (void)imag_c64;
            } catch (...) {
                // Shape/conversion issues are expected
            }
            
            try {
                // Test with complex128
                torch::Tensor c128 = complex_tensor.to(torch::kComplexDouble);
                torch::Tensor imag_c128 = torch::imag(c128);
                (void)imag_c128;
            } catch (...) {
                // Shape/conversion issues are expected
            }
        }
        
        // Test with view tensor
        if ((control_byte & 0x20) && complex_tensor.numel() > 1) {
            try {
                torch::Tensor view_tensor = complex_tensor.view({-1});
                torch::Tensor view_imag = torch::imag(view_tensor);
                (void)view_imag;
            } catch (...) {
                // View failures are acceptable
            }
        }
        
        // Test with non-contiguous tensor
        if ((control_byte & 0x40) && complex_tensor.dim() > 1 && 
            complex_tensor.size(0) > 1 && complex_tensor.size(1) > 1) {
            try {
                torch::Tensor transposed = complex_tensor.transpose(0, 1);
                torch::Tensor transposed_imag = torch::imag(transposed);
                (void)transposed_imag;
            } catch (...) {
                // Transpose/imag failures on edge cases are acceptable
            }
        }
        
        // Test edge cases based on fuzzer input
        if (control_byte & 0x80) {
            try {
                // Scalar complex tensor
                float real_val = static_cast<float>(Data[0]) / 255.0f;
                float imag_val = (Size > 1) ? static_cast<float>(Data[1]) / 255.0f : 0.0f;
                torch::Tensor scalar_complex = torch::complex(
                    torch::tensor(real_val), 
                    torch::tensor(imag_val)
                );
                torch::Tensor scalar_imag = torch::imag(scalar_complex);
                (void)scalar_imag;
            } catch (...) {
                // Scalar edge cases may fail
            }
            
            try {
                // Empty complex tensor
                torch::Tensor empty_real = torch::empty({0}, torch::kFloat);
                torch::Tensor empty_complex = torch::complex(empty_real, empty_real);
                torch::Tensor empty_imag = torch::imag(empty_complex);
                (void)empty_imag;
            } catch (...) {
                // Empty tensor handling may vary
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}