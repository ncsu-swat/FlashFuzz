#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal data to create a tensor
        if (Size < 3) {
            // Not enough data for even basic tensor creation
            return 0;
        }

        // Create primary tensor for erfc_ operation
        torch::Tensor tensor;
        try {
            tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If we can't create a basic tensor, try with minimal defaults
            if (offset < Size) {
                uint8_t dtype_selector = Data[offset % Size];
                auto dtype = fuzzer_utils::parseDataType(dtype_selector);
                
                // Create a small tensor with remaining data
                auto options = torch::TensorOptions().dtype(dtype);
                tensor = torch::randn({2, 2}, options);
            } else {
                return 0;
            }
        }

        // Explore different tensor configurations based on remaining data
        if (offset < Size) {
            uint8_t config_byte = Data[offset++];
            
            // Test various tensor properties
            if (config_byte & 0x01) {
                // Test with non-contiguous tensor
                if (tensor.dim() > 1 && tensor.size(0) > 1 && tensor.size(1) > 1) {
                    tensor = tensor.transpose(0, 1);
                }
            }
            
            if (config_byte & 0x02) {
                // Test with sliced tensor (creates non-contiguous views)
                if (tensor.numel() > 2) {
                    tensor = tensor.narrow(0, 0, std::max(int64_t(1), tensor.size(0) / 2));
                }
            }
            
            if (config_byte & 0x04) {
                // Test with reshaped tensor
                if (tensor.numel() > 0) {
                    tensor = tensor.reshape({-1});
                }
            }
            
            if (config_byte & 0x08) {
                // Test with tensor requiring gradient (if floating point)
                if (tensor.dtype() == torch::kFloat || tensor.dtype() == torch::kDouble ||
                    tensor.dtype() == torch::kHalf || tensor::dtype() == torch::kBFloat16) {
                    tensor.requires_grad_(true);
                }
            }
            
            if (config_byte & 0x10) {
                // Test with pinned memory (if CUDA available)
                if (torch::cuda::is_available() && offset < Size) {
                    uint8_t cuda_byte = Data[offset++];
                    if (cuda_byte & 0x01) {
                        try {
                            tensor = tensor.pin_memory();
                        } catch (...) {
                            // Ignore pinned memory errors
                        }
                    }
                }
            }
        }

        // Clone tensor for comparison if needed
        torch::Tensor original;
        bool should_compare = false;
        if (offset < Size && Data[offset++] & 0x01) {
            original = tensor.clone();
            should_compare = true;
        }

        // Apply erfc_ operation
        try {
            // erfc_ only works on floating point tensors
            if (tensor.dtype() == torch::kFloat || tensor.dtype() == torch::kDouble ||
                tensor.dtype() == torch::kHalf || tensor.dtype() == torch::kBFloat16 ||
                tensor.dtype() == torch::kComplexFloat || tensor.dtype() == torch::kComplexDouble) {
                
                tensor.erfc_();
                
                // Verify the operation succeeded
                if (tensor.numel() > 0) {
                    // Check for NaN/Inf in result
                    if (tensor.dtype() == torch::kFloat || tensor.dtype() == torch::kDouble) {
                        auto has_nan = torch::isnan(tensor).any().item<bool>();
                        auto has_inf = torch::isinf(tensor).any().item<bool>();
                        
                        // These are valid results for erfc, just note them
                        if (has_nan || has_inf) {
                            // Valid mathematical results, continue
                        }
                    }
                }
                
                // Additional validation: erfc should produce values in [0, 2] for real inputs
                if (should_compare && original.numel() > 0 && 
                    (original.dtype() == torch::kFloat || original.dtype() == torch::kDouble)) {
                    // For real numbers, erfc(x) is in range [0, 2]
                    auto min_val = tensor.min();
                    auto max_val = tensor.max();
                    
                    // Check bounds (with some tolerance for numerical errors)
                    if (min_val.item<double>() < -0.001 || max_val.item<double>() > 2.001) {
                        // Unexpected range, but don't crash - this might reveal bugs
                    }
                }
                
            } else {
                // Try to convert to float and then apply erfc_
                try {
                    tensor = tensor.to(torch::kFloat);
                    tensor.erfc_();
                } catch (const c10::Error& e) {
                    // Some dtypes might not be convertible, that's ok
                    return 0;
                }
            }
        } catch (const c10::Error& e) {
            // PyTorch errors are expected for invalid operations
            return 0;
        }

        // Test chained operations if more data available
        if (offset < Size) {
            uint8_t chain_ops = Data[offset++];
            
            try {
                if (chain_ops & 0x01) {
                    // Chain with another erfc_
                    tensor.erfc_();
                }
                
                if (chain_ops & 0x02) {
                    // Chain with arithmetic operation
                    tensor.add_(1.0);
                }
                
                if (chain_ops & 0x04) {
                    // Chain with multiplication
                    tensor.mul_(2.0);
                }
                
                if (chain_ops & 0x08) {
                    // Chain with clamp
                    tensor.clamp_(-10, 10);
                }
            } catch (const c10::Error& e) {
                // Chained operations might fail, that's ok
                return 0;
            }
        }

        // Test edge cases with special values if we have more data
        if (offset < Size && tensor.numel() > 0) {
            uint8_t special_val = Data[offset++];
            
            try {
                if (special_val & 0x01) {
                    // Test with infinity
                    if (tensor.dtype() == torch::kFloat || tensor.dtype() == torch::kDouble) {
                        tensor[0] = std::numeric_limits<float>::infinity();
                        tensor.erfc_();
                    }
                }
                
                if (special_val & 0x02) {
                    // Test with negative infinity
                    if (tensor.dtype() == torch::kFloat || tensor.dtype() == torch::kDouble) {
                        if (tensor.numel() > 1) {
                            tensor[1] = -std::numeric_limits<float>::infinity();
                            tensor.erfc_();
                        }
                    }
                }
                
                if (special_val & 0x04) {
                    // Test with NaN
                    if (tensor.dtype() == torch::kFloat || tensor.dtype() == torch::kDouble) {
                        if (tensor.numel() > 2) {
                            tensor[2] = std::numeric_limits<float>::quiet_NaN();
                            tensor.erfc_();
                        }
                    }
                }
            } catch (const c10::Error& e) {
                // Special value operations might fail
                return 0;
            }
        }

        // Test with different memory layouts if more data
        if (offset < Size && Size - offset >= 2) {
            try {
                // Create another tensor with different properties
                torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Try to make shapes compatible for binary operations
                if (tensor.numel() == tensor2.numel() && tensor.numel() > 0) {
                    tensor2 = tensor2.reshape(tensor.sizes());
                    
                    // Test erfc_ on the second tensor
                    if (tensor2.dtype() == torch::kFloat || tensor2.dtype() == torch::kDouble ||
                        tensor2.dtype() == torch::kHalf || tensor2.dtype() == torch::kBFloat16) {
                        tensor2.erfc_();
                    }
                }
            } catch (...) {
                // Second tensor creation might fail
            }
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}