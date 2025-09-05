#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <limits>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for configuration
        if (Size < 10) {
            return 0;  // Not enough data to construct meaningful test
        }

        // Parse configuration bytes
        uint8_t config_byte = Data[offset++];
        
        // Extract dimension for cosine similarity (can be negative)
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Constrain dim to reasonable range [-10, 10]
            dim = dim % 21 - 10;
        }
        
        // Extract epsilon value
        double eps = 1e-8;  // default
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Constrain epsilon to reasonable positive range
            eps = std::abs(eps);
            if (std::isnan(eps) || std::isinf(eps) || eps == 0.0) {
                eps = 1e-8;
            } else {
                // Constrain to range [1e-20, 1.0]
                eps = std::max(1e-20, std::min(1.0, eps));
            }
        }
        
        // Decide on test scenario based on config_byte
        uint8_t scenario = config_byte % 8;
        
        torch::Tensor x1, x2, result;
        
        switch (scenario) {
            case 0: {
                // Standard case: two tensors with same shape
                x1 = fuzzer_utils::createTensor(Data, Size, offset);
                x2 = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Ensure both tensors have same shape for this scenario
                if (x1.numel() > 0 && x2.numel() > 0 && x1.sizes() != x2.sizes()) {
                    x2 = x2.reshape(x1.sizes());
                }
                break;
            }
            case 1: {
                // Broadcasting case: different but compatible shapes
                x1 = fuzzer_utils::createTensor(Data, Size, offset);
                x2 = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Try to make shapes broadcastable by adjusting x2
                if (x1.dim() > 0 && x2.dim() > 0 && x1.dim() != x2.dim()) {
                    // Add or remove dimensions to make broadcasting possible
                    if (x1.dim() > x2.dim()) {
                        std::vector<int64_t> new_shape(x1.dim(), 1);
                        for (int i = 0; i < x2.dim(); ++i) {
                            new_shape[x1.dim() - x2.dim() + i] = x2.size(i);
                        }
                        x2 = x2.reshape(new_shape);
                    }
                }
                break;
            }
            case 2: {
                // Edge case: scalar tensors
                x1 = torch::tensor(3.14f);
                x2 = torch::tensor(2.71f);
                break;
            }
            case 3: {
                // Edge case: zero tensors
                auto shape = fuzzer_utils::parseShape(Data, offset, Size, 2);
                if (shape.empty()) shape = {3, 4};
                x1 = torch::zeros(shape);
                x2 = torch::zeros(shape);
                break;
            }
            case 4: {
                // Edge case: inf/nan values
                x1 = fuzzer_utils::createTensor(Data, Size, offset);
                x2 = fuzzer_utils::createTensor(Data, Size, offset);
                
                if (x1.numel() > 0) {
                    // Inject some inf/nan values
                    x1.view(-1)[0] = std::numeric_limits<float>::infinity();
                    if (x1.numel() > 1) {
                        x1.view(-1)[1] = std::numeric_limits<float>::quiet_NaN();
                    }
                }
                break;
            }
            case 5: {
                // Edge case: very large/small values
                x1 = fuzzer_utils::createTensor(Data, Size, offset);
                x2 = fuzzer_utils::createTensor(Data, Size, offset);
                
                if (x1.dtype() == torch::kFloat || x1.dtype() == torch::kDouble) {
                    x1 = x1 * 1e10;
                    x2 = x2 * 1e-10;
                }
                break;
            }
            case 6: {
                // Edge case: single dimension tensors
                int64_t size = (offset < Size) ? (Data[offset++] % 100 + 1) : 10;
                x1 = torch::randn({size});
                x2 = torch::randn({size});
                dim = 0;  // Only valid dimension for 1D tensors
                break;
            }
            case 7: {
                // Mixed dtypes (will test type promotion)
                x1 = fuzzer_utils::createTensor(Data, Size, offset);
                x2 = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Convert x2 to different dtype if they're the same
                if (x1.dtype() == x2.dtype() && x1.numel() > 0) {
                    if (x1.dtype() == torch::kFloat) {
                        x2 = x2.to(torch::kDouble);
                    } else if (x1.dtype() == torch::kDouble) {
                        x2 = x2.to(torch::kFloat);
                    }
                }
                break;
            }
        }
        
        // Ensure tensors are floating point for cosine similarity
        if (!x1.is_floating_point()) {
            x1 = x1.to(torch::kFloat);
        }
        if (!x2.is_floating_point()) {
            x2 = x2.to(torch::kFloat);
        }
        
        // Perform cosine similarity computation
        try {
            result = torch::cosine_similarity(x1, x2, dim, eps);
            
            // Validate result properties
            if (result.defined()) {
                // Check that output has one fewer dimension (unless inputs were scalar)
                if (x1.dim() > 0 && x2.dim() > 0) {
                    auto broadcast_shape = torch::broadcast_shapes({x1.sizes().vec(), x2.sizes().vec()});
                    int64_t expected_dim = broadcast_shape.size() - 1;
                    if (expected_dim < 0) expected_dim = 0;
                    
                    // Cosine similarity should be in range [-1, 1]
                    if (result.numel() > 0 && !result.has_names()) {
                        auto min_val = result.min().item<double>();
                        auto max_val = result.max().item<double>();
                        
                        // Allow some tolerance for numerical errors
                        if (min_val < -1.1 || max_val > 1.1) {
                            std::cerr << "Warning: Cosine similarity out of expected range: [" 
                                     << min_val << ", " << max_val << "]" << std::endl;
                        }
                    }
                }
                
                // Test gradient computation if tensors require grad
                if (x1.requires_grad() || x2.requires_grad()) {
                    try {
                        auto sum_result = result.sum();
                        sum_result.backward();
                    } catch (...) {
                        // Gradient computation failed, but that's okay for fuzzing
                    }
                }
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for invalid inputs
            #ifdef DEBUG_FUZZ
            std::cerr << "PyTorch error in cosine_similarity: " << e.what() << std::endl;
            #endif
        }
        
        // Additional edge case: try with different memory layouts
        if (x1.numel() > 1 && x1.dim() > 1) {
            try {
                auto x1_transposed = x1.transpose(0, -1);
                auto x2_transposed = x2.dim() > 1 ? x2.transpose(0, -1) : x2;
                auto result2 = torch::cosine_similarity(x1_transposed, x2_transposed, dim, eps);
            } catch (...) {
                // Expected for some configurations
            }
        }
        
        // Test with contiguous vs non-contiguous tensors
        if (x1.numel() > 0 && !x1.is_contiguous()) {
            try {
                auto x1_cont = x1.contiguous();
                auto result3 = torch::cosine_similarity(x1_cont, x2, dim, eps);
            } catch (...) {
                // Expected for some configurations
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