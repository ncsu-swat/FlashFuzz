#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstring>        // For memcpy

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need minimum data for tensor creation
        if (Size < 4) {
            return -1;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // nan_to_num_ only works meaningfully on floating point tensors
        if (!input_tensor.is_floating_point()) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Parse replacement values for nan, posinf, neginf
        c10::optional<double> nan_value = c10::nullopt;
        c10::optional<double> posinf_value = c10::nullopt;
        c10::optional<double> neginf_value = c10::nullopt;
        
        // Parse nan replacement if we have enough data
        if (offset + sizeof(double) <= Size) {
            double val;
            std::memcpy(&val, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Only use finite values as replacements
            if (std::isfinite(val)) {
                nan_value = val;
            }
        }
        
        // Parse posinf replacement if we have enough data
        if (offset + sizeof(double) <= Size) {
            double val;
            std::memcpy(&val, Data + offset, sizeof(double));
            offset += sizeof(double);
            if (std::isfinite(val)) {
                posinf_value = val;
            }
        }
        
        // Parse neginf replacement if we have enough data
        if (offset + sizeof(double) <= Size) {
            double val;
            std::memcpy(&val, Data + offset, sizeof(double));
            offset += sizeof(double);
            if (std::isfinite(val)) {
                neginf_value = val;
            }
        }
        
        // Make a copy of the input tensor to verify the in-place operation
        torch::Tensor input_copy = input_tensor.clone();
        
        // Apply nan_to_num_ in-place operation
        input_tensor.nan_to_num_(nan_value, posinf_value, neginf_value);
        
        // Verify the operation by comparing with non-in-place version
        torch::Tensor expected = input_copy.nan_to_num(nan_value, posinf_value, neginf_value);
        
        // Check if the in-place operation produced the expected result
        try {
            if (!torch::allclose(input_tensor, expected, 1e-5, 1e-8)) {
                // Mismatch - this could indicate a bug
                std::cerr << "nan_to_num_ produced unexpected result" << std::endl;
            }
        } catch (...) {
            // allclose may fail for certain tensor states, ignore
        }
        
        // Test with default parameters (all nullopt)
        torch::Tensor default_test = input_copy.clone();
        default_test.nan_to_num_();
        
        // Test with only nan replacement
        torch::Tensor nan_only_test = input_copy.clone();
        nan_only_test.nan_to_num_(0.0, c10::nullopt, c10::nullopt);
        
        // Test with nan and posinf replacement
        torch::Tensor nan_posinf_test = input_copy.clone();
        nan_posinf_test.nan_to_num_(0.0, 1e10, c10::nullopt);
        
        // Test with double precision tensor
        try {
            torch::Tensor double_tensor = input_copy.to(torch::kDouble);
            double_tensor.nan_to_num_(nan_value, posinf_value, neginf_value);
        } catch (...) {
            // Conversion may fail for some inputs
        }
        
        // Test with half precision tensor if available
        try {
            torch::Tensor half_tensor = input_copy.to(torch::kHalf);
            half_tensor.nan_to_num_();
        } catch (...) {
            // Half precision may not be supported
        }
        
        // Test with empty tensor
        try {
            torch::Tensor empty_tensor = torch::empty({0}, torch::kFloat);
            empty_tensor.nan_to_num_();
        } catch (...) {
            // Empty tensor handling may vary
        }
        
        // Test with tensor containing actual NaN and Inf values
        try {
            torch::Tensor special_tensor = torch::tensor({
                std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                1.0f, -1.0f, 0.0f
            });
            special_tensor.nan_to_num_(0.0, 1e10, -1e10);
        } catch (...) {
            // May fail on some configurations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}