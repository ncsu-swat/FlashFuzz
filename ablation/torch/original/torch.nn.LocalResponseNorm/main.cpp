#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        if (Size < 10) {
            // Need minimum bytes for parameters and tensor metadata
            return 0;
        }

        size_t offset = 0;

        // Parse LocalResponseNorm parameters from fuzzer input
        // 1. Parse size parameter (amount of neighbouring channels)
        int64_t lrn_size = 2; // default
        if (offset + sizeof(uint8_t) <= Size) {
            uint8_t size_byte = Data[offset++];
            // Map to reasonable range [1, 20] for size parameter
            lrn_size = 1 + (size_byte % 20);
        }

        // 2. Parse alpha parameter
        double alpha = 0.0001; // default
        if (offset + sizeof(uint32_t) <= Size) {
            uint32_t alpha_raw;
            std::memcpy(&alpha_raw, Data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            // Map to reasonable range [1e-6, 1.0]
            alpha = 1e-6 + (alpha_raw % 1000000) / 1000000.0;
        }

        // 3. Parse beta parameter
        double beta = 0.75; // default
        if (offset + sizeof(uint32_t) <= Size) {
            uint32_t beta_raw;
            std::memcpy(&beta_raw, Data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            // Map to reasonable range [0.01, 2.0]
            beta = 0.01 + (beta_raw % 200) / 100.0;
        }

        // 4. Parse k parameter
        double k = 1.0; // default
        if (offset + sizeof(uint32_t) <= Size) {
            uint32_t k_raw;
            std::memcpy(&k_raw, Data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            // Map to reasonable range [0.0, 10.0]
            k = (k_raw % 1000) / 100.0;
        }

        // Create input tensor from remaining data
        torch::Tensor input_tensor;
        
        // Try to create a tensor with fuzzer-controlled shape
        if (offset < Size) {
            try {
                input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // LocalResponseNorm requires at least 2 dimensions (N, C, ...)
                // Reshape if necessary to ensure valid input
                if (input_tensor.dim() < 2) {
                    // Reshape scalar or 1D tensor to have batch and channel dims
                    auto numel = input_tensor.numel();
                    if (numel == 0) {
                        // Create a minimal valid tensor
                        input_tensor = torch::randn({1, 1});
                    } else if (input_tensor.dim() == 0) {
                        // Scalar -> [1, 1]
                        input_tensor = input_tensor.reshape({1, 1});
                    } else if (input_tensor.dim() == 1) {
                        // 1D -> [1, C]
                        input_tensor = input_tensor.reshape({1, input_tensor.size(0)});
                    }
                }
                
            } catch (const std::exception& e) {
                // If tensor creation fails, create a default tensor
                input_tensor = torch::randn({2, 3, 4, 4});
            }
        } else {
            // No data left for tensor, create default
            input_tensor = torch::randn({2, 3, 4, 4});
        }

        // Ensure tensor is floating point (LocalResponseNorm requires float types)
        if (!input_tensor.is_floating_point()) {
            input_tensor = input_tensor.to(torch::kFloat32);
        }

        // Create LocalResponseNorm module with parsed parameters
        torch::nn::LocalResponseNorm lrn_module(
            torch::nn::LocalResponseNormOptions(lrn_size)
                .alpha(alpha)
                .beta(beta)
                .k(k)
        );

        // Test with different input variations
        
        // 1. Test with original tensor
        try {
            auto output = lrn_module->forward(input_tensor);
            
            // Verify output shape matches input
            if (output.sizes() != input_tensor.sizes()) {
                std::cerr << "Shape mismatch: input " << input_tensor.sizes() 
                         << " vs output " << output.sizes() << std::endl;
            }
            
            // Check for NaN/Inf in output
            if (torch::any(torch::isnan(output)).item<bool>()) {
                std::cerr << "NaN detected in output" << std::endl;
            }
            if (torch::any(torch::isinf(output)).item<bool>()) {
                std::cerr << "Inf detected in output" << std::endl;
            }
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors
            std::cerr << "PyTorch error with original tensor: " << e.what() << std::endl;
        }

        // 2. Test with contiguous version
        if (!input_tensor.is_contiguous()) {
            try {
                auto contiguous_input = input_tensor.contiguous();
                auto output = lrn_module->forward(contiguous_input);
            } catch (const c10::Error& e) {
                std::cerr << "PyTorch error with contiguous tensor: " << e.what() << std::endl;
            }
        }

        // 3. Test with different memory formats if applicable
        if (input_tensor.dim() == 4) {
            try {
                // Try channels_last format for 4D tensors
                auto channels_last_input = input_tensor.to(torch::MemoryFormat::ChannelsLast);
                auto output = lrn_module->forward(channels_last_input);
            } catch (const c10::Error& e) {
                std::cerr << "PyTorch error with channels_last format: " << e.what() << std::endl;
            }
        }

        // 4. Test with requires_grad
        if (input_tensor.dtype() == torch::kFloat32 || input_tensor.dtype() == torch::kFloat64) {
            try {
                auto grad_input = input_tensor.requires_grad_(true);
                auto output = lrn_module->forward(grad_input);
                
                // Try backward pass
                if (output.requires_grad()) {
                    auto grad_output = torch::ones_like(output);
                    output.backward(grad_output);
                }
            } catch (const c10::Error& e) {
                std::cerr << "PyTorch error with gradient computation: " << e.what() << std::endl;
            }
        }

        // 5. Test edge cases with extreme dimensions
        try {
            // Test with single channel (edge case for normalization)
            if (input_tensor.size(1) > 1) {
                auto single_channel = input_tensor.narrow(1, 0, 1);
                auto output = lrn_module->forward(single_channel);
            }
            
            // Test with many channels if tensor is large enough
            if (input_tensor.dim() >= 2 && input_tensor.size(0) > 0) {
                auto many_channels = torch::randn({input_tensor.size(0), 100, 4, 4}, 
                                                 input_tensor.options());
                auto output = lrn_module->forward(many_channels);
            }
        } catch (const c10::Error& e) {
            std::cerr << "PyTorch error with edge case dimensions: " << e.what() << std::endl;
        }

        // 6. Test with different batch sizes
        if (input_tensor.dim() >= 2 && input_tensor.size(0) > 1) {
            try {
                // Test with batch size 1
                auto single_batch = input_tensor.narrow(0, 0, 1);
                auto output = lrn_module->forward(single_batch);
            } catch (const c10::Error& e) {
                std::cerr << "PyTorch error with single batch: " << e.what() << std::endl;
            }
        }

        // 7. Test with higher dimensional inputs (5D, 6D)
        try {
            if (input_tensor.dim() == 4) {
                // Extend to 5D
                auto input_5d = input_tensor.unsqueeze(-1);
                auto output = lrn_module->forward(input_5d);
            }
        } catch (const c10::Error& e) {
            std::cerr << "PyTorch error with 5D input: " << e.what() << std::endl;
        }

        // 8. Test with zero/negative values in input
        try {
            auto zero_tensor = torch::zeros_like(input_tensor);
            auto output_zero = lrn_module->forward(zero_tensor);
            
            auto negative_tensor = -torch::abs(input_tensor);
            auto output_neg = lrn_module->forward(negative_tensor);
        } catch (const c10::Error& e) {
            std::cerr << "PyTorch error with zero/negative inputs: " << e.what() << std::endl;
        }

        // 9. Test parameter edge cases by creating new modules
        try {
            // Test with size = 1 (minimal neighborhood)
            torch::nn::LocalResponseNorm lrn_size1(
                torch::nn::LocalResponseNormOptions(1).alpha(alpha).beta(beta).k(k)
            );
            lrn_size1->forward(input_tensor);
            
            // Test with very large size
            int64_t large_size = std::min(static_cast<int64_t>(100), 
                                         input_tensor.dim() >= 2 ? input_tensor.size(1) : 1);
            torch::nn::LocalResponseNorm lrn_large(
                torch::nn::LocalResponseNormOptions(large_size).alpha(alpha).beta(beta).k(k)
            );
            lrn_large->forward(input_tensor);
        } catch (const c10::Error& e) {
            std::cerr << "PyTorch error with parameter edge cases: " << e.what() << std::endl;
        }

        // 10. Test with extreme parameter values
        try {
            torch::nn::LocalResponseNorm lrn_extreme(
                torch::nn::LocalResponseNormOptions(lrn_size)
                    .alpha(1e-10)  // Very small alpha
                    .beta(10.0)    // Large beta
                    .k(0.0)        // Zero k
            );
            lrn_extreme->forward(input_tensor);
        } catch (const c10::Error& e) {
            std::cerr << "PyTorch error with extreme parameters: " << e.what() << std::endl;
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}