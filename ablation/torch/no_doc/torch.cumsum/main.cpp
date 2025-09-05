#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for tensor creation and operation parameters
        if (Size < 4) {
            return 0;  // Not enough data, but keep for coverage
        }

        // Create primary tensor from fuzzer input
        torch::Tensor tensor;
        try {
            tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception &e) {
            // If tensor creation fails, try with minimal tensor
            tensor = torch::randn({1});
            offset = Size;  // Mark all data consumed
        }

        // Parse dimension for cumsum operation
        int64_t dim = 0;
        if (offset < Size) {
            uint8_t dim_byte = Data[offset++];
            // Map to valid dimension range based on tensor rank
            if (tensor.dim() > 0) {
                dim = static_cast<int64_t>(dim_byte) % tensor.dim();
                // Also test negative dimensions for coverage
                if (offset < Size && Data[offset++] % 2 == 0) {
                    dim = -tensor.dim() + (dim_byte % tensor.dim());
                }
            }
        }

        // Parse dtype for output (optional parameter)
        c10::optional<torch::ScalarType> dtype = c10::nullopt;
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            // 25% chance to specify output dtype
            if (dtype_selector % 4 == 0 && offset < Size) {
                dtype = fuzzer_utils::parseDataType(Data[offset++]);
            }
        }

        // Test various tensor states for broader coverage
        if (offset < Size) {
            uint8_t state_selector = Data[offset++];
            switch (state_selector % 8) {
                case 0:
                    // Test with non-contiguous tensor
                    if (tensor.dim() >= 2 && tensor.size(0) > 1 && tensor.size(1) > 1) {
                        tensor = tensor.transpose(0, 1);
                    }
                    break;
                case 1:
                    // Test with sliced tensor
                    if (tensor.numel() > 2) {
                        tensor = tensor.narrow(0, 0, std::max(int64_t(1), tensor.size(0) / 2));
                    }
                    break;
                case 2:
                    // Test with view
                    if (tensor.numel() > 1 && tensor.dim() == 1) {
                        tensor = tensor.view({-1, 1});
                        dim = dim % tensor.dim();  // Adjust dim for new shape
                    }
                    break;
                case 3:
                    // Test with expanded tensor
                    if (tensor.dim() >= 1 && tensor.size(0) == 1) {
                        tensor = tensor.expand({3, -1});
                        dim = dim % tensor.dim();
                    }
                    break;
                case 4:
                    // Test with permuted tensor
                    if (tensor.dim() >= 2) {
                        std::vector<int64_t> perm;
                        for (int64_t i = tensor.dim() - 1; i >= 0; --i) {
                            perm.push_back(i);
                        }
                        tensor = tensor.permute(perm);
                    }
                    break;
                case 5:
                    // Test with requires_grad
                    if (tensor.dtype() == torch::kFloat || tensor.dtype() == torch::kDouble ||
                        tensor.dtype() == torch::kHalf || tensor.dtype() == torch::kBFloat16) {
                        tensor = tensor.requires_grad_(true);
                    }
                    break;
                case 6:
                    // Test with sparse tensor (if applicable)
                    if (tensor.dim() == 2 && tensor.numel() > 0 && 
                        (tensor.dtype() == torch::kFloat || tensor.dtype() == torch::kDouble)) {
                        try {
                            tensor = tensor.to_sparse();
                        } catch (...) {
                            // Keep dense if sparse conversion fails
                        }
                    }
                    break;
                case 7:
                    // Test with zero-strided dimension
                    if (tensor.dim() >= 2 && tensor.size(1) == 1) {
                        tensor = tensor.expand({tensor.size(0), 3});
                    }
                    break;
            }
        }

        // Main operation: torch.cumsum
        torch::Tensor result;
        try {
            if (dtype.has_value()) {
                result = torch::cumsum(tensor, dim, dtype.value());
            } else {
                result = torch::cumsum(tensor, dim);
            }
            
            // Verify result properties for coverage
            if (result.defined()) {
                // Check shape preservation
                if (result.sizes() != tensor.sizes()) {
                    std::cerr << "Shape mismatch after cumsum" << std::endl;
                }
                
                // Additional operations to increase coverage
                if (offset < Size && Data[offset++] % 3 == 0) {
                    // Test backward pass if applicable
                    if (result.requires_grad()) {
                        try {
                            auto grad_out = torch::ones_like(result);
                            result.backward(grad_out);
                        } catch (...) {
                            // Backward might fail for some dtypes
                        }
                    }
                    
                    // Test in-place variant if exists
                    try {
                        auto tensor_copy = tensor.clone();
                        tensor_copy.cumsum_(dim);
                    } catch (...) {
                        // In-place might not be supported for all cases
                    }
                }
                
                // Test edge cases
                if (offset < Size) {
                    uint8_t edge_selector = Data[offset++];
                    switch (edge_selector % 4) {
                        case 0:
                            // Test with NaN/Inf values if floating point
                            if (tensor.is_floating_point() && tensor.numel() > 0) {
                                auto test_tensor = tensor.clone();
                                test_tensor[0] = std::numeric_limits<float>::quiet_NaN();
                                try {
                                    auto nan_result = torch::cumsum(test_tensor, dim);
                                } catch (...) {}
                            }
                            break;
                        case 1:
                            // Test with very large values
                            if (tensor.dtype() == torch::kInt64 && tensor.numel() > 0) {
                                auto test_tensor = tensor.clone();
                                test_tensor.fill_(std::numeric_limits<int64_t>::max() / 2);
                                try {
                                    auto overflow_result = torch::cumsum(test_tensor, dim);
                                } catch (...) {}
                            }
                            break;
                        case 2:
                            // Test cumsum on result (double cumsum)
                            try {
                                auto double_cumsum = torch::cumsum(result, dim);
                            } catch (...) {}
                            break;
                        case 3:
                            // Test with different memory formats
                            if (tensor.dim() == 4) {
                                try {
                                    auto channels_last = tensor.to(torch::MemoryFormat::ChannelsLast);
                                    auto cl_result = torch::cumsum(channels_last, dim);
                                } catch (...) {}
                            }
                            break;
                    }
                }
            }
        } catch (const c10::Error &e) {
            // PyTorch-specific errors are expected for invalid operations
            return 0;
        } catch (const std::exception &e) {
            // Log unexpected exceptions but continue
            std::cerr << "Unexpected exception in cumsum: " << e.what() << std::endl;
            return 0;
        }

        // Test cumsum with different dimensions if tensor is multi-dimensional
        if (tensor.dim() > 1 && offset < Size) {
            for (int64_t d = 0; d < tensor.dim() && offset < Size; ++d) {
                try {
                    auto result_d = torch::cumsum(tensor, d);
                    // Also try negative dimension indexing
                    auto result_neg = torch::cumsum(tensor, d - tensor.dim());
                } catch (...) {
                    // Some dimensions might fail
                }
                offset++;  // Consume a byte per iteration
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