#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for two tensors
        if (Size < 4) {
            // Not enough data for even basic tensor metadata
            return 0;
        }

        // Create first tensor (will be modified in-place)
        torch::Tensor tensor1;
        try {
            tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception &e) {
            // If we can't create the first tensor, try with minimal valid tensor
            tensor1 = torch::ones({1}, torch::kInt32);
        }
        
        // Create second tensor
        torch::Tensor tensor2;
        if (offset < Size) {
            try {
                tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            } catch (const std::exception &e) {
                // If we can't create second tensor, create one with compatible shape
                tensor2 = torch::ones_like(tensor1);
            }
        } else {
            // No more data, create tensor with same shape as tensor1
            tensor2 = torch::ones_like(tensor1);
        }
        
        // lcm_ requires integer types, so convert if needed
        bool need_conversion = false;
        if (!tensor1.dtype().isIntegral(false)) {  // false = don't include bool
            need_conversion = true;
            // Try different integer types based on remaining fuzzer data
            if (offset < Size) {
                uint8_t type_selector = Data[offset++] % 5;
                switch(type_selector) {
                    case 0: tensor1 = tensor1.to(torch::kInt8); break;
                    case 1: tensor1 = tensor1.to(torch::kInt16); break;
                    case 2: tensor1 = tensor1.to(torch::kInt32); break;
                    case 3: tensor1 = tensor1.to(torch::kInt64); break;
                    case 4: tensor1 = tensor1.to(torch::kUInt8); break;
                    default: tensor1 = tensor1.to(torch::kInt32); break;
                }
            } else {
                tensor1 = tensor1.to(torch::kInt32);
            }
        }
        
        if (!tensor2.dtype().isIntegral(false)) {
            // Convert tensor2 to match tensor1's dtype for compatibility
            tensor2 = tensor2.to(tensor1.dtype());
        }
        
        // Handle broadcasting by trying different shape configurations
        if (offset < Size) {
            uint8_t broadcast_mode = Data[offset++] % 6;
            switch(broadcast_mode) {
                case 0:
                    // Keep as is - test normal broadcasting
                    break;
                case 1:
                    // Make tensor2 scalar
                    if (tensor2.numel() > 0) {
                        tensor2 = tensor2.flatten()[0];
                    }
                    break;
                case 2:
                    // Make tensor1 and tensor2 have same shape
                    if (tensor1.numel() > 0 && tensor2.numel() > 0) {
                        tensor2 = tensor2.reshape_as(tensor1);
                    }
                    break;
                case 3:
                    // Add dimension to tensor2
                    tensor2 = tensor2.unsqueeze(0);
                    break;
                case 4:
                    // Transpose if 2D
                    if (tensor2.dim() == 2) {
                        tensor2 = tensor2.t();
                    }
                    break;
                case 5:
                    // Make tensor2 have broadcastable shape
                    if (tensor1.dim() > 0) {
                        auto shape = std::vector<int64_t>(tensor1.dim(), 1);
                        if (tensor2.numel() > 0) {
                            shape[tensor1.dim() - 1] = std::min(tensor2.numel(), tensor1.size(-1));
                            tensor2 = tensor2.flatten().slice(0, 0, shape[tensor1.dim() - 1]).reshape(shape);
                        }
                    }
                    break;
            }
        }
        
        // Test different memory layouts
        if (offset < Size) {
            uint8_t layout_mode = Data[offset++] % 4;
            switch(layout_mode) {
                case 0:
                    // Keep contiguous
                    break;
                case 1:
                    // Make non-contiguous via transpose
                    if (tensor1.dim() >= 2) {
                        tensor1 = tensor1.transpose(0, 1).transpose(0, 1);
                    }
                    break;
                case 2:
                    // Make non-contiguous via slice
                    if (tensor1.size(0) > 1) {
                        tensor1 = tensor1.slice(0, 0, tensor1.size(0), 2);
                    }
                    break;
                case 3:
                    // Make tensor2 non-contiguous
                    if (tensor2.dim() >= 2) {
                        tensor2 = tensor2.transpose(0, 1).transpose(0, 1);
                    }
                    break;
            }
        }
        
        // Clone tensor1 before in-place operation for potential validation
        torch::Tensor tensor1_original = tensor1.clone();
        
        // Call lcm_ (in-place operation)
        try {
            tensor1.lcm_(tensor2);
            
            // Validate result properties
            // Check that tensor1 was actually modified (unless inputs were special)
            if (tensor1.numel() > 0 && tensor2.numel() > 0) {
                // Basic sanity checks
                if (tensor1.dtype() != tensor1_original.dtype()) {
                    std::cerr << "Warning: dtype changed after lcm_" << std::endl;
                }
                if (tensor1.sizes() != tensor1_original.sizes()) {
                    std::cerr << "Warning: shape changed after lcm_" << std::endl;
                }
                
                // For small tensors, verify lcm properties
                if (tensor1.numel() <= 10 && tensor1.numel() == tensor2.numel()) {
                    auto t1_flat = tensor1_original.flatten();
                    auto t2_flat = tensor2.flatten();
                    auto result_flat = tensor1.flatten();
                    
                    for (int64_t i = 0; i < tensor1.numel(); ++i) {
                        // LCM should be divisible by both inputs (when non-zero)
                        int64_t a = t1_flat[i].item<int64_t>();
                        int64_t b = t2_flat[i].item<int64_t>();
                        int64_t lcm_val = result_flat[i].item<int64_t>();
                        
                        if (a != 0 && b != 0 && lcm_val != 0) {
                            if (lcm_val % a != 0 || lcm_val % b != 0) {
                                std::cerr << "LCM property violation at index " << i << std::endl;
                            }
                        }
                    }
                }
            }
            
        } catch (const c10::Error &e) {
            // PyTorch-specific errors (shape mismatch, dtype issues, etc.)
            // These are expected for many inputs, just continue
            return 0;
        } catch (const std::runtime_error &e) {
            // Runtime errors from broadcasting or other issues
            return 0;
        }
        
        // Test edge cases with special values
        if (offset < Size && Data[offset++] % 2 == 0) {
            try {
                // Test with zeros
                torch::Tensor zero_tensor = torch::zeros_like(tensor1_original);
                zero_tensor.lcm_(tensor2);
                
                // Test with ones  
                torch::Tensor ones_tensor = torch::ones_like(tensor1_original);
                ones_tensor.lcm_(tensor2);
                
                // Test with negative values (if signed type)
                if (tensor1_original.dtype() != torch::kUInt8) {
                    torch::Tensor neg_tensor = -torch::ones_like(tensor1_original);
                    neg_tensor.lcm_(tensor2);
                }
            } catch (...) {
                // Ignore errors in edge case testing
            }
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}