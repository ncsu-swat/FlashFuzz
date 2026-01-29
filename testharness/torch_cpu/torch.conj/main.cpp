#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Skip if there's not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor from the fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test with the original tensor (could be any dtype)
        torch::Tensor result = torch::conj(input_tensor);
        
        // Force computation by accessing data
        if (result.defined() && result.numel() > 0) {
            // Use sum() instead of item() to handle multi-element tensors
            auto sum_val = result.sum();
            (void)sum_val;
        }
        
        // Test conjugate view method
        torch::Tensor conj_view = input_tensor.conj();
        if (conj_view.defined() && conj_view.numel() > 0) {
            auto sum_val = conj_view.sum();
            (void)sum_val;
        }
        
        // Test is_conj flag
        bool is_conj = result.is_conj();
        (void)is_conj;
        
        // Test resolve_conj to materialize the conjugate
        torch::Tensor resolved = torch::resolve_conj(result);
        if (resolved.defined() && resolved.numel() > 0) {
            auto sum_val = resolved.sum();
            (void)sum_val;
        }
        
        // Also test with explicitly complex tensor for better coverage
        // conj is most meaningful for complex numbers
        if (offset < Size) {
            try {
                // Create a complex tensor from the input
                torch::Tensor real_part = fuzzer_utils::createTensor(Data, Size, offset);
                torch::Tensor imag_part = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Make sure shapes match for complex construction
                if (real_part.sizes() == imag_part.sizes() && 
                    real_part.numel() > 0 && imag_part.numel() > 0) {
                    // Convert to float for complex construction
                    torch::Tensor real_float = real_part.to(torch::kFloat);
                    torch::Tensor imag_float = imag_part.to(torch::kFloat);
                    
                    // Create complex tensor
                    torch::Tensor complex_tensor = torch::complex(real_float, imag_float);
                    
                    // Test conj on complex tensor
                    torch::Tensor complex_conj = torch::conj(complex_tensor);
                    
                    if (complex_conj.defined() && complex_conj.numel() > 0) {
                        // Access real and imaginary parts
                        torch::Tensor real_result = torch::real(complex_conj);
                        torch::Tensor imag_result = torch::imag(complex_conj);
                        auto r_sum = real_result.sum();
                        auto i_sum = imag_result.sum();
                        (void)r_sum;
                        (void)i_sum;
                    }
                    
                    // Test conj_physical on complex tensor
                    torch::Tensor conj_phys = torch::conj_physical(complex_tensor);
                    if (conj_phys.defined() && conj_phys.numel() > 0) {
                        auto sum_val = torch::real(conj_phys).sum();
                        (void)sum_val;
                    }
                }
            } catch (...) {
                // Inner catch: silently ignore shape mismatches or dtype issues
            }
        }
        
        // Test conj_physical on original tensor
        try {
            torch::Tensor conj_phys = torch::conj_physical(input_tensor);
            if (conj_phys.defined() && conj_phys.numel() > 0) {
                auto sum_val = conj_phys.sum();
                (void)sum_val;
            }
        } catch (...) {
            // Inner catch: silently ignore if conj_physical fails for certain dtypes
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}