#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test various math functions from torch namespace
        try {
            // Basic math operations
            torch::Tensor abs_result = torch::abs(input);
            torch::Tensor sqrt_result = torch::sqrt(input.abs());
            torch::Tensor sin_result = torch::sin(input);
            torch::Tensor cos_result = torch::cos(input);
            torch::Tensor tan_result = torch::tan(input);
            
            // Exponential and logarithmic functions
            torch::Tensor exp_result = torch::exp(input);
            torch::Tensor log_result = torch::log(input.abs() + 1e-10);
            torch::Tensor log10_result = torch::log10(input.abs() + 1e-10);
            torch::Tensor log2_result = torch::log2(input.abs() + 1e-10);
            
            // Power functions
            torch::Tensor pow_result = torch::pow(input.abs(), 2.5);
            
            // Rounding functions
            torch::Tensor ceil_result = torch::ceil(input);
            torch::Tensor floor_result = torch::floor(input);
            torch::Tensor round_result = torch::round(input);
            torch::Tensor trunc_result = torch::trunc(input);
            
            // Hyperbolic functions
            torch::Tensor sinh_result = torch::sinh(input);
            torch::Tensor cosh_result = torch::cosh(input);
            torch::Tensor tanh_result = torch::tanh(input);
            
            // Inverse trigonometric functions
            torch::Tensor asin_result = torch::asin(input.clamp(-1, 1));
            torch::Tensor acos_result = torch::acos(input.clamp(-1, 1));
            torch::Tensor atan_result = torch::atan(input);
            
            // Special functions
            torch::Tensor erf_result = torch::erf(input);
            torch::Tensor erfc_result = torch::erfc(input);
            torch::Tensor erfinv_result = torch::erfinv(input.clamp(-0.99, 0.99));
            
            // Complex number functions if applicable
            if (input.is_complex()) {
                torch::Tensor real_result = torch::real(input);
                torch::Tensor imag_result = torch::imag(input);
                torch::Tensor angle_result = torch::angle(input);
            }
            
            // Test some math operations that take multiple tensors
            if (offset + 2 < Size) {
                torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Try binary operations
                torch::Tensor atan2_result = torch::atan2(input, input2);
                torch::Tensor hypot_result = torch::hypot(input, input2);
                torch::Tensor fmod_result = torch::fmod(input, input2 + 1e-10);
                torch::Tensor remainder_result = torch::remainder(input, input2 + 1e-10);
            }
            
            // Test available special math functions
            torch::Tensor special_digamma_result = torch::special::digamma(input.abs() + 1);
        } catch (const std::exception &e) {
            // Catch exceptions from math operations but continue testing
        }
        
        // Test matrix math functions if tensor has at least 2 dimensions
        if (input.dim() >= 2) {
            try {
                torch::Tensor det_result = torch::det(input);
                torch::Tensor trace_result = torch::trace(input);
                
                // Try matrix decompositions
                try {
                    auto qr_result = torch::qr(input);
                } catch (...) {}
                
                try {
                    auto svd_result = torch::svd(input);
                } catch (...) {}
                
                try {
                    auto eig_result = torch::eig(input);
                } catch (...) {}
                
                // Matrix inverse
                try {
                    torch::Tensor inv_result = torch::inverse(input);
                } catch (...) {}
            } catch (const std::exception &e) {
                // Catch exceptions from matrix operations but continue testing
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