#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 4) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Basic math operations
        try {
            torch::Tensor abs_result = torch::abs(input);
            torch::Tensor sqrt_result = torch::sqrt(torch::abs(input));
            torch::Tensor sin_result = torch::sin(input);
            torch::Tensor cos_result = torch::cos(input);
            torch::Tensor tan_result = torch::tan(input);
        } catch (...) {}
        
        // Exponential and logarithmic functions
        try {
            torch::Tensor exp_result = torch::exp(input);
            torch::Tensor log_result = torch::log(torch::abs(input) + 1e-10);
            torch::Tensor log10_result = torch::log10(torch::abs(input) + 1e-10);
            torch::Tensor log2_result = torch::log2(torch::abs(input) + 1e-10);
        } catch (...) {}
        
        // Power functions
        try {
            torch::Tensor pow_result = torch::pow(torch::abs(input) + 1e-10, 2.5);
        } catch (...) {}
        
        // Rounding functions
        try {
            torch::Tensor ceil_result = torch::ceil(input);
            torch::Tensor floor_result = torch::floor(input);
            torch::Tensor round_result = torch::round(input);
            torch::Tensor trunc_result = torch::trunc(input);
        } catch (...) {}
        
        // Hyperbolic functions
        try {
            torch::Tensor sinh_result = torch::sinh(input);
            torch::Tensor cosh_result = torch::cosh(input);
            torch::Tensor tanh_result = torch::tanh(input);
        } catch (...) {}
        
        // Inverse trigonometric functions
        try {
            torch::Tensor clamped = input.clamp(-1.0, 1.0);
            torch::Tensor asin_result = torch::asin(clamped);
            torch::Tensor acos_result = torch::acos(clamped);
            torch::Tensor atan_result = torch::atan(input);
        } catch (...) {}
        
        // Special functions
        try {
            torch::Tensor erf_result = torch::erf(input);
            torch::Tensor erfc_result = torch::erfc(input);
            torch::Tensor erfinv_result = torch::erfinv(input.clamp(-0.99, 0.99));
        } catch (...) {}
        
        // Gamma functions
        try {
            torch::Tensor lgamma_result = torch::lgamma(torch::abs(input) + 1);
            torch::Tensor digamma_result = torch::digamma(torch::abs(input) + 1);
        } catch (...) {}
        
        // Test binary math operations with a second tensor
        if (offset < Size) {
            try {
                torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Ensure same shape for binary operations
                if (input.sizes() == input2.sizes()) {
                    try {
                        torch::Tensor atan2_result = torch::atan2(input, input2);
                    } catch (...) {}
                    
                    try {
                        torch::Tensor hypot_result = torch::hypot(input, input2);
                    } catch (...) {}
                    
                    try {
                        torch::Tensor fmod_result = torch::fmod(input, input2.abs() + 1e-10);
                    } catch (...) {}
                    
                    try {
                        torch::Tensor remainder_result = torch::remainder(input, input2.abs() + 1e-10);
                    } catch (...) {}
                }
            } catch (...) {}
        }
        
        // Test matrix math functions if tensor has at least 2 dimensions
        if (input.dim() >= 2 && input.size(-1) == input.size(-2) && input.size(-1) > 0) {
            // Matrix must be square for these operations
            try {
                torch::Tensor det_result = torch::det(input);
            } catch (...) {}
            
            try {
                torch::Tensor trace_result = torch::trace(input);
            } catch (...) {}
            
            // Matrix inverse (requires non-singular matrix)
            try {
                torch::Tensor inv_result = torch::inverse(input);
            } catch (...) {}
            
            // QR decomposition using torch::qr (deprecated but available in C++)
            try {
                auto qr_result = torch::qr(input);
            } catch (...) {}
            
            // SVD decomposition using torch::svd
            try {
                auto svd_result = torch::svd(input);
            } catch (...) {}
        }
        
        // Additional element-wise operations
        try {
            torch::Tensor neg_result = torch::neg(input);
            torch::Tensor sign_result = torch::sign(input);
            torch::Tensor reciprocal_result = torch::reciprocal(input.abs() + 1e-10);
        } catch (...) {}
        
        // Bit operations for integer tensors
        if (input.is_floating_point()) {
            try {
                torch::Tensor frac_result = torch::frac(input);
            } catch (...) {}
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}