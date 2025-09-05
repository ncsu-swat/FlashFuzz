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
        
        // Need at least minimal bytes for tensor creation
        if (Size < 3) {
            // Still try to create a minimal tensor
            torch::Tensor t = torch::zeros({1});
            auto result = torch::asin(t);
            return 0;
        }

        // Create primary tensor from fuzzer input
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If tensor creation fails, create a fallback tensor with remaining bytes
            if (Size > 0) {
                uint8_t rank = Data[0] % 3; // 0, 1, or 2D
                std::vector<int64_t> shape;
                if (rank == 0) {
                    shape = {};
                } else if (rank == 1) {
                    shape = {static_cast<int64_t>(1 + (Data[0] % 10))};
                } else {
                    shape = {2, 3};
                }
                
                // Use different dtypes based on available data
                torch::ScalarType dtype = torch::kFloat;
                if (Size > 1) {
                    uint8_t dtype_selector = Data[1] % 6;
                    switch(dtype_selector) {
                        case 0: dtype = torch::kFloat; break;
                        case 1: dtype = torch::kDouble; break;
                        case 2: dtype = torch::kHalf; break;
                        case 3: dtype = torch::kBFloat16; break;
                        case 4: dtype = torch::kComplexFloat; break;
                        case 5: dtype = torch::kComplexDouble; break;
                    }
                }
                
                auto options = torch::TensorOptions().dtype(dtype);
                input_tensor = torch::randn(shape, options);
            } else {
                input_tensor = torch::tensor(0.5f);
            }
        }

        // Apply asin operation - main target
        torch::Tensor result;
        try {
            result = torch::asin(input_tensor);
            
            // Verify result properties
            if (result.defined()) {
                // Check shape preservation
                if (result.sizes() != input_tensor.sizes()) {
                    std::cerr << "Shape mismatch after asin" << std::endl;
                }
                
                // Check dtype preservation (asin should preserve dtype for floating types)
                if (input_tensor.is_floating_point() && result.dtype() != input_tensor.dtype()) {
                    std::cerr << "Dtype changed unexpectedly" << std::endl;
                }
                
                // Access some elements to trigger potential issues
                if (result.numel() > 0) {
                    auto flat = result.flatten();
                    if (flat.numel() > 0) {
                        flat[0].item<float>();  // May throw for non-float types
                    }
                }
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific errors (e.g., complex number issues, NaN handling)
            // Continue execution - these are expected for edge cases
        }

        // Test with edge case values if we have more data
        if (offset < Size && Size - offset >= 2) {
            uint8_t edge_selector = Data[offset++] % 8;
            torch::Tensor edge_tensor;
            
            switch(edge_selector) {
                case 0: // Values exactly at boundaries
                    edge_tensor = torch::tensor({-1.0f, 0.0f, 1.0f});
                    break;
                case 1: // Values slightly outside valid range
                    edge_tensor = torch::tensor({-1.001f, 1.001f});
                    break;
                case 2: // NaN and Inf
                    edge_tensor = torch::tensor({std::numeric_limits<float>::quiet_NaN(), 
                                                 std::numeric_limits<float>::infinity(),
                                                 -std::numeric_limits<float>::infinity()});
                    break;
                case 3: // Very small values
                    edge_tensor = torch::tensor({std::numeric_limits<float>::min(),
                                                 -std::numeric_limits<float>::min(),
                                                 std::numeric_limits<float>::denorm_min()});
                    break;
                case 4: // Empty tensor
                    edge_tensor = torch::empty({0});
                    break;
                case 5: // Large tensor with uniform values
                    edge_tensor = torch::full({100, 100}, 0.5f);
                    break;
                case 6: // Complex numbers (if supported)
                    edge_tensor = torch::complex(torch::tensor(0.5f), torch::tensor(0.3f));
                    break;
                case 7: // Integer tensor (will be converted)
                    edge_tensor = torch::tensor({-1, 0, 1}, torch::kInt32);
                    break;
            }
            
            try {
                auto edge_result = torch::asin(edge_tensor);
                
                // Additional operations on result
                if (edge_result.defined() && edge_result.numel() > 0) {
                    // Test gradient computation if applicable
                    if (edge_result.requires_grad()) {
                        auto sum = edge_result.sum();
                        sum.backward();
                    }
                    
                    // Test in-place operation
                    if (edge_tensor.is_floating_point()) {
                        edge_tensor.asin_();
                    }
                }
            } catch (const c10::Error& e) {
                // Expected for some edge cases
            }
        }

        // Test with different memory layouts if more data available
        if (offset < Size && Size - offset >= 1) {
            uint8_t layout_selector = Data[offset++] % 4;
            torch::Tensor layout_tensor = torch::rand({4, 4});
            
            switch(layout_selector) {
                case 0: // Transposed (non-contiguous)
                    layout_tensor = layout_tensor.t();
                    break;
                case 1: // Sliced (non-contiguous)
                    layout_tensor = torch::rand({10, 10}).slice(0, 1, 8, 2);
                    break;
                case 2: // Permuted dimensions
                    layout_tensor = torch::rand({2, 3, 4}).permute({2, 0, 1});
                    break;
                case 3: // View with different shape
                    layout_tensor = torch::rand({12}).view({3, 4});
                    break;
            }
            
            try {
                auto layout_result = torch::asin(layout_tensor);
                // Verify contiguity is handled correctly
                if (layout_tensor.is_contiguous() != layout_result.is_contiguous()) {
                    // This might be expected behavior
                }
            } catch (const c10::Error& e) {
                // Continue on errors
            }
        }

        // Test with requires_grad if we have a floating point tensor
        if (input_tensor.is_floating_point() && input_tensor.numel() > 0) {
            try {
                auto grad_tensor = input_tensor.clone().requires_grad_(true);
                auto grad_result = torch::asin(grad_tensor);
                if (grad_result.requires_grad() && grad_result.numel() > 0) {
                    auto loss = grad_result.sum();
                    loss.backward();
                    // Access gradient to ensure it's computed
                    if (grad_tensor.grad().defined()) {
                        grad_tensor.grad().sum();
                    }
                }
            } catch (const c10::Error& e) {
                // Gradient computation might fail for certain values
            }
        }

        // Test batch processing with different shapes
        if (offset < Size) {
            try {
                std::vector<torch::Tensor> batch_tensors;
                uint8_t batch_size = (Size > offset) ? (Data[offset] % 5 + 1) : 2;
                
                for (uint8_t i = 0; i < batch_size; ++i) {
                    batch_tensors.push_back(torch::rand({2, 3}) * 2.0 - 1.0); // Range [-1, 1]
                }
                
                auto stacked = torch::stack(batch_tensors);
                auto batch_result = torch::asin(stacked);
                
                // Verify batch dimension is preserved
                if (batch_result.size(0) != batch_size) {
                    std::cerr << "Batch dimension not preserved" << std::endl;
                }
            } catch (const c10::Error& e) {
                // Batch operations might fail
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}