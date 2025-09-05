#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 10) {  // Minimal size for basic tensor creation
        return 0;
    }

    try {
        size_t offset = 0;
        
        // Parse control bytes for operation variants
        uint8_t transpose_q = (offset < size) ? data[offset++] : 0;
        bool transpose = transpose_q % 2;
        
        // Create input tensor (needs to be at least 2D for householder_product)
        torch::Tensor input;
        if (offset < size) {
            try {
                input = fuzzer_utils::createTensor(data, size, offset);
                
                // Ensure input is at least 2D
                if (input.dim() < 2) {
                    // Reshape to 2D if needed
                    auto numel = input.numel();
                    if (numel > 0) {
                        int64_t dim1 = std::max(int64_t(1), int64_t(std::sqrt(numel)));
                        int64_t dim2 = numel / dim1;
                        if (dim1 * dim2 != numel) {
                            dim2 = numel;
                            dim1 = 1;
                        }
                        input = input.reshape({dim1, dim2});
                    } else {
                        input = input.reshape({0, 0});
                    }
                }
            } catch (...) {
                // Create default input if parsing fails
                input = torch::randn({3, 4}, torch::kFloat32);
            }
        } else {
            input = torch::randn({3, 4}, torch::kFloat32);
        }
        
        // Create tau tensor (Householder coefficients)
        torch::Tensor tau;
        if (offset < size) {
            try {
                tau = fuzzer_utils::createTensor(data, size, offset);
                
                // tau should be 1D or 2D
                if (tau.dim() > 2) {
                    // Flatten to 1D if too many dimensions
                    tau = tau.flatten();
                } else if (tau.dim() == 0) {
                    // Make it 1D with single element
                    tau = tau.reshape({1});
                }
                
                // Ensure tau has compatible size with input
                // The number of reflections should not exceed min(m, n) where input is m x n
                int64_t max_reflections = std::min(input.size(-2), input.size(-1));
                if (tau.dim() == 1 && tau.size(0) > max_reflections) {
                    tau = tau.slice(0, 0, max_reflections);
                } else if (tau.dim() == 2 && tau.size(0) > max_reflections) {
                    tau = tau.slice(0, 0, max_reflections);
                }
                
                // Ensure tau is not empty
                if (tau.numel() == 0) {
                    tau = torch::randn({1}, input.options());
                }
            } catch (...) {
                // Create default tau if parsing fails
                int64_t num_reflections = std::min(input.size(-2), input.size(-1));
                tau = torch::randn({num_reflections}, input.options());
            }
        } else {
            // Default tau based on input dimensions
            int64_t num_reflections = std::min(input.size(-2), input.size(-1));
            tau = torch::randn({num_reflections}, input.options());
        }
        
        // Ensure compatible dtypes
        if (input.dtype() != tau.dtype()) {
            // Convert tau to match input dtype if they differ
            if (input.is_floating_point() && tau.is_floating_point()) {
                tau = tau.to(input.dtype());
            } else if (!input.is_floating_point() && tau.is_floating_point()) {
                // Convert input to float if it's not already
                input = input.to(torch::kFloat32);
                tau = tau.to(torch::kFloat32);
            } else if (input.is_floating_point() && !tau.is_floating_point()) {
                tau = tau.to(input.dtype());
            } else {
                // Both are non-floating, convert both to float
                input = input.to(torch::kFloat32);
                tau = tau.to(torch::kFloat32);
            }
        }
        
        // Ensure input is floating point (required for householder_product)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
            tau = tau.to(torch::kFloat32);
        }
        
        // Try different device configurations
        uint8_t device_config = (offset < size) ? data[offset++] : 0;
        if (device_config % 4 == 1 && torch::cuda::is_available()) {
            try {
                input = input.cuda();
                tau = tau.cuda();
            } catch (...) {
                // Keep on CPU if CUDA transfer fails
            }
        }
        
        // Test with different memory layouts
        uint8_t layout_config = (offset < size) ? data[offset++] : 0;
        if (layout_config % 3 == 1 && input.is_contiguous()) {
            try {
                input = input.transpose(-2, -1).contiguous().transpose(-2, -1);
            } catch (...) {}
        } else if (layout_config % 3 == 2) {
            try {
                input = input.contiguous();
                tau = tau.contiguous();
            } catch (...) {}
        }
        
        // Call householder_product with different configurations
        try {
            torch::Tensor result = torch::linalg::householder_product(input, tau);
            
            // Verify result properties
            if (result.dim() != input.dim()) {
                std::cerr << "Unexpected dimension change in result" << std::endl;
            }
            
            // Check for NaN/Inf
            if (result.has_nan().item<bool>() || result.has_inf().item<bool>()) {
                // This is acceptable for edge cases but log it
                #ifdef DEBUG_FUZZ
                std::cout << "Result contains NaN or Inf" << std::endl;
                #endif
            }
            
            // Try with transposed version if we have enough data
            if (transpose) {
                try {
                    // Create a new input that's compatible with transposed operation
                    torch::Tensor input_t = input.clone();
                    if (input_t.dim() >= 2) {
                        input_t = input_t.transpose(-2, -1);
                        // Adjust tau dimensions if needed for transposed operation
                        int64_t new_max_reflections = std::min(input_t.size(-2), input_t.size(-1));
                        torch::Tensor tau_t = tau.clone();
                        if (tau_t.dim() == 1 && tau_t.size(0) > new_max_reflections) {
                            tau_t = tau_t.slice(0, 0, new_max_reflections);
                        }
                        torch::Tensor result_t = torch::linalg::householder_product(input_t, tau_t);
                    }
                } catch (const std::exception& e) {
                    // Transposed operation might fail for certain shapes
                    #ifdef DEBUG_FUZZ
                    std::cout << "Transposed operation failed: " << e.what() << std::endl;
                    #endif
                }
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for invalid inputs
            #ifdef DEBUG_FUZZ
            std::cout << "PyTorch error in householder_product: " << e.what() << std::endl;
            #endif
        }
        
        // Test edge cases with different tau shapes
        if (offset < size && data[offset++] % 2 == 0) {
            try {
                // Test with 2D tau (batch mode)
                if (tau.dim() == 1) {
                    tau = tau.unsqueeze(1);
                    torch::Tensor result_batch = torch::linalg::householder_product(input, tau);
                }
            } catch (...) {
                // Batch mode might not be supported for all configurations
            }
        }
        
        // Test with zero-sized tensors
        if (offset < size && data[offset++] % 10 == 0) {
            try {
                torch::Tensor zero_input = torch::empty({0, 3}, input.options());
                torch::Tensor zero_tau = torch::empty({0}, tau.options());
                torch::Tensor zero_result = torch::linalg::householder_product(zero_input, zero_tau);
            } catch (...) {
                // Zero-sized tensors might cause issues
            }
        }
        
    } catch (const std::bad_alloc& e) {
        // Memory allocation failure - this is a valid fuzzing outcome
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // Catch any other unexpected exceptions
        std::cout << "Exception caught: Unknown exception" << std::endl;
        return -1;
    }
    
    return 0;
}