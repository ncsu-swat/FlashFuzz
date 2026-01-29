#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::abs, std::isnan, std::isinf

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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract eps parameter if we have more data
        double eps = 1e-6; // Default value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure eps is within a reasonable range (avoid NaN/Inf)
            if (std::isnan(eps) || std::isinf(eps)) {
                eps = 1e-6;
            }
            eps = std::abs(eps);
            if (eps == 0.0 || eps > 1.0) {
                eps = 1e-6;
            }
        }
        
        // logit requires floating point input
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat);
        }
        
        // Apply logit operation with default eps (no second argument)
        torch::Tensor result1 = torch::logit(input);
        
        // Apply logit operation with custom eps
        torch::Tensor result2 = torch::logit(input, eps);
        
        // Apply in-place logit operation (no eps argument for default)
        try {
            torch::Tensor input_copy = input.clone();
            input_copy.logit_();
        } catch (...) {
            // Silently handle in-place operation failures
        }
        
        // Apply in-place logit operation with custom eps
        try {
            torch::Tensor input_copy2 = input.clone();
            input_copy2.logit_(eps);
        } catch (...) {
            // Silently handle in-place operation failures
        }
        
        // Try with different dtypes
        try {
            torch::Tensor float_input = input.to(torch::kFloat);
            torch::Tensor result_float = torch::logit(float_input, eps);
        } catch (...) {
            // Silently handle dtype conversion failures
        }
        
        try {
            torch::Tensor double_input = input.to(torch::kDouble);
            torch::Tensor result_double = torch::logit(double_input, eps);
        } catch (...) {
            // Silently handle dtype conversion failures
        }
        
        // Test with clamped input (logit expects values in (0, 1) range)
        try {
            torch::Tensor clamped = torch::clamp(input, eps, 1.0 - eps);
            torch::Tensor result_clamped = torch::logit(clamped, eps);
        } catch (...) {
            // Silently handle clamping failures
        }
        
        // Test with different eps values from fuzzer data
        if (offset < Size) {
            double eps_small = static_cast<double>(Data[offset] % 10 + 1) * 1e-8;
            try {
                torch::Tensor result_small_eps = torch::logit(input, eps_small);
            } catch (...) {
                // Silently handle failures with different eps
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