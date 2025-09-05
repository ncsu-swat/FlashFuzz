#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for configuration
        if (Size < 4) {
            return 0;
        }

        // Parse configuration parameters from fuzzer input
        uint8_t num_bits = Data[offset++];
        num_bits = (num_bits % 8) + 1; // Range: 1-8 bits for quantization
        
        uint8_t num_bins = Data[offset++]; 
        int numel = 1 + (num_bins % 200); // Range: 1-200 bins
        
        uint8_t use_symmetric = Data[offset++];
        bool symmetric = (use_symmetric % 2) == 0;
        
        uint8_t preserve_sparsity_byte = Data[offset++];
        bool preserve_sparsity = (preserve_sparsity_byte % 2) == 0;
        
        // Create input tensor from fuzzer data
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If tensor creation fails, try with a simple random tensor
            if (offset < Size) {
                uint8_t rank = Data[offset++] % 4 + 1;
                std::vector<int64_t> shape;
                for (int i = 0; i < rank && offset < Size; i++) {
                    shape.push_back((Data[offset++] % 10) + 1);
                }
                if (shape.empty()) shape = {10};
                input = torch::randn(shape, torch::kFloat32);
            } else {
                input = torch::randn({10, 10}, torch::kFloat32);
            }
        }
        
        // Ensure tensor is float type (required for quantization)
        if (input.scalar_type() != torch::kFloat32 && 
            input.scalar_type() != torch::kFloat64 &&
            input.scalar_type() != torch::kFloat16 &&
            input.scalar_type() != torch::kBFloat16) {
            input = input.to(torch::kFloat32);
        }
        
        // Flatten tensor if needed (choose_qparams_optimized typically works on flattened tensors)
        torch::Tensor flattened = input.flatten();
        
        // Handle edge cases for tensor content
        if (offset < Size) {
            uint8_t edge_case = Data[offset++];
            switch (edge_case % 8) {
                case 0: // All zeros
                    flattened.zero_();
                    break;
                case 1: // All ones
                    flattened.fill_(1.0);
                    break;
                case 2: // Infinity values
                    if (flattened.numel() > 0) {
                        flattened[0] = std::numeric_limits<float>::infinity();
                        if (flattened.numel() > 1) {
                            flattened[1] = -std::numeric_limits<float>::infinity();
                        }
                    }
                    break;
                case 3: // NaN values
                    if (flattened.numel() > 0) {
                        flattened[0] = std::numeric_limits<float>::quiet_NaN();
                    }
                    break;
                case 4: // Very large values
                    flattened.mul_(1e10);
                    break;
                case 5: // Very small values
                    flattened.mul_(1e-10);
                    break;
                case 6: // Mixed positive/negative
                    for (int64_t i = 0; i < flattened.numel(); i += 2) {
                        if (i < flattened.numel()) {
                            flattened[i] = -flattened[i].item<float>();
                        }
                    }
                    break;
                default:
                    // Keep original values
                    break;
            }
        }
        
        // Call choose_qparams_optimized with various configurations
        try {
            // Standard call
            auto [scale1, zero_point1] = torch::choose_qparams_optimized(
                flattened, 
                numel,
                num_bits,
                symmetric,
                preserve_sparsity
            );
            
            // Try with different tensor views
            if (input.dim() > 1) {
                torch::Tensor transposed = input.t();
                auto flat_t = transposed.flatten();
                auto [scale2, zero_point2] = torch::choose_qparams_optimized(
                    flat_t,
                    numel,
                    num_bits,
                    symmetric,
                    preserve_sparsity
                );
            }
            
            // Try with contiguous and non-contiguous tensors
            if (!flattened.is_contiguous()) {
                auto [scale3, zero_point3] = torch::choose_qparams_optimized(
                    flattened,
                    numel,
                    num_bits,
                    symmetric,
                    preserve_sparsity
                );
            }
            
            torch::Tensor contiguous = flattened.contiguous();
            auto [scale4, zero_point4] = torch::choose_qparams_optimized(
                contiguous,
                numel,
                num_bits,
                symmetric,
                preserve_sparsity
            );
            
            // Try with empty tensor
            if (offset < Size && Data[offset++] % 10 == 0) {
                torch::Tensor empty_tensor = torch::empty({0}, torch::kFloat32);
                auto [scale5, zero_point5] = torch::choose_qparams_optimized(
                    empty_tensor,
                    1,
                    num_bits,
                    symmetric,
                    preserve_sparsity
                );
            }
            
            // Try with single element tensor
            if (offset < Size && Data[offset++] % 10 == 1) {
                torch::Tensor single = torch::tensor({1.0f});
                auto [scale6, zero_point6] = torch::choose_qparams_optimized(
                    single,
                    numel,
                    num_bits,
                    symmetric,
                    preserve_sparsity
                );
            }
            
            // Try with different numel values
            for (int i = 0; i < 3 && offset < Size; i++) {
                int new_numel = 1 + (Data[offset++] % flattened.numel());
                auto [scale7, zero_point7] = torch::choose_qparams_optimized(
                    flattened,
                    new_numel,
                    num_bits,
                    symmetric,
                    preserve_sparsity
                );
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for edge cases
            return 0;
        } catch (const std::runtime_error& e) {
            // Runtime errors might be expected for certain inputs
            return 0;
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}