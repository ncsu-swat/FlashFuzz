#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::abs, std::fmod

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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create first tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor with remaining data
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for a second tensor, clone the first one
            tensor2 = tensor1.clone();
            
            // Optionally modify tensor2 slightly to test different scenarios
            if (tensor2.numel() > 0 && tensor2.is_floating_point()) {
                // Add a small value to make tensors slightly different
                tensor2.add_(0.001);
            }
        }
        
        // Extract rtol and atol parameters from remaining data
        double rtol = 1e-5;  // default value
        double atol = 1e-8;  // default value
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&rtol, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure rtol is positive and not too large
            if (!std::isfinite(rtol)) {
                rtol = 1e-5;
            } else {
                rtol = std::abs(rtol);
                rtol = std::fmod(rtol, 1.0) + 1e-9;
            }
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&atol, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure atol is positive and not too large
            if (!std::isfinite(atol)) {
                atol = 1e-8;
            } else {
                atol = std::abs(atol);
                atol = std::fmod(atol, 1.0) + 1e-9;
            }
        }
        
        // Test torch::allclose with various parameter combinations
        // Wrap in try-catch since tensors may have incompatible shapes/dtypes
        try {
            bool result1 = torch::allclose(tensor1, tensor2);
            (void)result1;
        } catch (const std::exception &) {
            // Shape or dtype mismatch - expected
        }
        
        try {
            // Test with explicit rtol and atol
            bool result2 = torch::allclose(tensor1, tensor2, rtol, atol);
            (void)result2;
        } catch (const std::exception &) {
            // Shape or dtype mismatch - expected
        }
        
        try {
            // Test with equal_nan=true
            bool result3 = torch::allclose(tensor1, tensor2, rtol, atol, true);
            (void)result3;
        } catch (const std::exception &) {
            // Shape or dtype mismatch - expected
        }
        
        try {
            // Test with swapped tensors
            bool result4 = torch::allclose(tensor2, tensor1, rtol, atol);
            (void)result4;
        } catch (const std::exception &) {
            // Shape or dtype mismatch - expected
        }
        
        // Test allclose with the same tensor (should always be true)
        bool result5 = torch::allclose(tensor1, tensor1);
        (void)result5;
        
        // Test with modified versions of the tensors
        if (tensor1.numel() > 0 && tensor1.is_floating_point()) {
            // Create a tensor with NaN values
            torch::Tensor tensor_with_nan = tensor1.clone().to(torch::kFloat);
            tensor_with_nan.flatten()[0] = std::numeric_limits<float>::quiet_NaN();
            
            try {
                // Test allclose with NaN values
                bool result_with_nan1 = torch::allclose(tensor1.to(torch::kFloat), tensor_with_nan);
                bool result_with_nan2 = torch::allclose(tensor1.to(torch::kFloat), tensor_with_nan, rtol, atol, true);
                (void)result_with_nan1;
                (void)result_with_nan2;
            } catch (const std::exception &) {
                // Conversion or comparison may fail
            }
            
            // Create a tensor with infinity values
            torch::Tensor tensor_with_inf = tensor1.clone().to(torch::kFloat);
            tensor_with_inf.flatten()[0] = std::numeric_limits<float>::infinity();
            
            try {
                // Test allclose with infinity values
                bool result_with_inf = torch::allclose(tensor1.to(torch::kFloat), tensor_with_inf);
                (void)result_with_inf;
            } catch (const std::exception &) {
                // Conversion or comparison may fail
            }
        }
        
        // Test with tensors of different shapes (expect failure)
        if (offset < Size && Size - offset > 2) {
            try {
                torch::Tensor tensor3 = fuzzer_utils::createTensor(Data, Size, offset);
                if (tensor3.sizes() != tensor1.sizes()) {
                    bool result_diff_shape = torch::allclose(tensor1, tensor3);
                    (void)result_diff_shape;
                }
            } catch (const std::exception &) {
                // Ignore exceptions from creating tensor3 or comparing different shapes
            }
        }
        
        // Test with tensors of different dtypes
        if (tensor1.dtype() != torch::kBool && tensor2.dtype() != torch::kBool) {
            try {
                torch::Tensor tensor1_float = tensor1.to(torch::kFloat);
                torch::Tensor tensor2_double = tensor2.to(torch::kDouble);
                bool result_diff_dtype = torch::allclose(tensor1_float, tensor2_double);
                (void)result_diff_dtype;
            } catch (const std::exception &) {
                // Ignore exceptions from dtype conversion or comparison
            }
        }
        
        // Test with zero-element tensors
        try {
            torch::Tensor empty1 = torch::empty({0});
            torch::Tensor empty2 = torch::empty({0});
            bool result_empty = torch::allclose(empty1, empty2);
            (void)result_empty;
        } catch (const std::exception &) {
            // May throw on empty tensors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}