#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for parsing
        if (Size < 10) {
            return 0;
        }

        // Parse input tensor - this will be the main data tensor for batch norm
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have remaining bytes for additional parameters
        if (offset >= Size) {
            // Still try with what we have
            auto result = torch::batch_norm_update_stats(input, torch::Tensor(), torch::Tensor(), 0.1);
            return 0;
        }

        // Parse momentum value from remaining bytes
        double momentum = 0.1; // default
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Clamp momentum to reasonable range [0, 1]
            momentum = std::abs(momentum);
            momentum = momentum - std::floor(momentum); // Keep in [0, 1)
        }

        // Try to parse running_mean tensor if we have enough data
        torch::Tensor running_mean;
        if (offset < Size && (Size - offset) > 2) {
            try {
                running_mean = fuzzer_utils::createTensor(Data, Size, offset);
                // Ensure running_mean has compatible shape with input's channel dimension
                if (input.dim() >= 2) {
                    int64_t num_features = input.size(1);
                    if (running_mean.numel() != num_features) {
                        // Reshape or create new tensor with correct size
                        running_mean = torch::zeros({num_features}, input.options());
                    }
                }
            } catch (...) {
                // If parsing fails, leave as undefined tensor
            }
        }

        // Try to parse running_var tensor if we have enough data
        torch::Tensor running_var;
        if (offset < Size && (Size - offset) > 2) {
            try {
                running_var = fuzzer_utils::createTensor(Data, Size, offset);
                // Ensure running_var has compatible shape
                if (input.dim() >= 2) {
                    int64_t num_features = input.size(1);
                    if (running_var.numel() != num_features) {
                        // Reshape or create new tensor with correct size
                        running_var = torch::ones({num_features}, input.options());
                    }
                }
            } catch (...) {
                // If parsing fails, leave as undefined tensor
            }
        }

        // Call batch_norm_update_stats with various parameter combinations
        try {
            // Basic call with just input
            auto [mean1, var1] = torch::batch_norm_update_stats(input, torch::Tensor(), torch::Tensor(), momentum);
            
            // Call with running_mean but no running_var
            if (running_mean.defined()) {
                auto [mean2, var2] = torch::batch_norm_update_stats(input, running_mean, torch::Tensor(), momentum);
            }
            
            // Call with running_var but no running_mean
            if (running_var.defined()) {
                auto [mean3, var3] = torch::batch_norm_update_stats(input, torch::Tensor(), running_var, momentum);
            }
            
            // Call with both running_mean and running_var
            if (running_mean.defined() && running_var.defined()) {
                auto [mean4, var4] = torch::batch_norm_update_stats(input, running_mean, running_var, momentum);
            }
            
            // Try with edge case momentum values
            auto [mean5, var5] = torch::batch_norm_update_stats(input, torch::Tensor(), torch::Tensor(), 0.0);
            auto [mean6, var6] = torch::batch_norm_update_stats(input, torch::Tensor(), torch::Tensor(), 1.0);
            
            // Try with transposed/permuted input to test different memory layouts
            if (input.dim() >= 2) {
                torch::Tensor permuted = input.permute({1, 0});
                for (int64_t i = 2; i < input.dim(); ++i) {
                    permuted = permuted.unsqueeze(-1);
                }
                auto [mean7, var7] = torch::batch_norm_update_stats(permuted, torch::Tensor(), torch::Tensor(), momentum);
            }
            
            // Try with non-contiguous tensors
            if (input.numel() > 1 && input.dim() > 0) {
                torch::Tensor strided = input.as_strided(
                    {input.size(0)}, 
                    {input.stride(0) > 0 ? input.stride(0) : 1}
                );
                auto [mean8, var8] = torch::batch_norm_update_stats(strided, torch::Tensor(), torch::Tensor(), momentum);
            }
            
            // Try with different device types if available
            if (torch::cuda::is_available() && offset < Size && Data[offset % Size] % 4 == 0) {
                torch::Tensor cuda_input = input.to(torch::kCUDA);
                torch::Tensor cuda_mean = running_mean.defined() ? running_mean.to(torch::kCUDA) : torch::Tensor();
                torch::Tensor cuda_var = running_var.defined() ? running_var.to(torch::kCUDA) : torch::Tensor();
                auto [mean9, var9] = torch::batch_norm_update_stats(cuda_input, cuda_mean, cuda_var, momentum);
            }
            
            // Try with requires_grad to test autograd paths
            if (input.is_floating_point()) {
                torch::Tensor grad_input = input.requires_grad_(true);
                auto [mean10, var10] = torch::batch_norm_update_stats(grad_input, torch::Tensor(), torch::Tensor(), momentum);
                
                // Trigger backward if output requires grad
                if (mean10.requires_grad()) {
                    mean10.sum().backward();
                }
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for invalid inputs
            // Continue fuzzing
        } catch (const std::runtime_error& e) {
            // Runtime errors from shape mismatches etc are expected
            // Continue fuzzing  
        }

        return 0;
    }
    catch (const std::bad_alloc& e)
    {
        // Memory allocation failures - discard this input
        return -1;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}