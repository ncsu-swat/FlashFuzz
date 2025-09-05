#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstdint>
#include <vector>
#include <limits>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for tensor creation
        if (Size < 2) {
            return 0;
        }

        // Create primary tensor for erfc_ operation
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Clone original for comparison if needed
        torch::Tensor original = tensor.clone();
        
        // Try different tensor configurations to maximize coverage
        if (offset < Size) {
            uint8_t config_byte = Data[offset++];
            
            // Test with different memory layouts
            if (config_byte & 0x01) {
                // Make non-contiguous by transposing if possible
                if (tensor.dim() >= 2) {
                    tensor = tensor.transpose(0, 1);
                }
            }
            
            // Test with requires_grad for floating point tensors
            if ((config_byte & 0x02) && tensor.is_floating_point()) {
                tensor.requires_grad_(true);
            }
            
            // Test with different device pinning
            if (config_byte & 0x04) {
                tensor = tensor.pin_memory();
            }
            
            // Test with view/reshape to stress memory layout
            if ((config_byte & 0x08) && tensor.numel() > 0) {
                try {
                    tensor = tensor.view({-1});
                    // Reshape back to original if we have shape info
                    if (original.sizes().size() > 0) {
                        tensor = tensor.view(original.sizes());
                    }
                } catch (...) {
                    // View might fail, continue with original tensor
                }
            }
        }
        
        // Apply erfc_ operation - this modifies tensor in-place
        try {
            tensor.erfc_();
            
            // Verify the operation worked by checking some properties
            if (tensor.is_floating_point() && tensor.numel() > 0) {
                // erfc values should be in range [0, 2] for real inputs
                // Check a few elements to ensure operation completed
                if (tensor.is_contiguous()) {
                    auto data_ptr = tensor.data_ptr();
                    if (tensor.scalar_type() == torch::kFloat) {
                        float* float_data = static_cast<float*>(data_ptr);
                        // Just access first element to ensure no segfault
                        volatile float first = float_data[0];
                        (void)first;
                    } else if (tensor.scalar_type() == torch::kDouble) {
                        double* double_data = static_cast<double*>(data_ptr);
                        volatile double first = double_data[0];
                        (void)first;
                    }
                }
                
                // Test gradient computation if applicable
                if (tensor.requires_grad()) {
                    try {
                        auto sum = tensor.sum();
                        sum.backward();
                    } catch (...) {
                        // Gradient computation might fail for some configurations
                    }
                }
            }
            
            // Additional operations to stress test the modified tensor
            if (offset < Size) {
                uint8_t extra_ops = Data[offset++];
                
                if (extra_ops & 0x01) {
                    // Chain another in-place operation
                    try {
                        tensor.neg_();
                    } catch (...) {}
                }
                
                if (extra_ops & 0x02) {
                    // Test cloning after modification
                    auto cloned = tensor.clone();
                    (void)cloned;
                }
                
                if (extra_ops & 0x04) {
                    // Test conversion to different dtype
                    try {
                        if (tensor.scalar_type() != torch::kDouble) {
                            tensor = tensor.to(torch::kDouble);
                        } else {
                            tensor = tensor.to(torch::kFloat);
                        }
                    } catch (...) {}
                }
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors - these are expected for invalid operations
            // Continue fuzzing
            return 0;
        } catch (const std::runtime_error& e) {
            // Runtime errors might indicate real issues
            // Log but continue
            return 0;
        }
        
        // Test edge cases with special tensors if we have more data
        if (offset + 2 < Size) {
            uint8_t edge_case = Data[offset++];
            
            try {
                torch::Tensor edge_tensor;
                
                switch (edge_case % 8) {
                    case 0:
                        // Empty tensor
                        edge_tensor = torch::empty({0}, torch::kFloat);
                        break;
                    case 1:
                        // Scalar tensor
                        edge_tensor = torch::tensor(3.14f);
                        break;
                    case 2:
                        // Tensor with inf values
                        edge_tensor = torch::full({2, 2}, std::numeric_limits<float>::infinity());
                        break;
                    case 3:
                        // Tensor with -inf values
                        edge_tensor = torch::full({2, 2}, -std::numeric_limits<float>::infinity());
                        break;
                    case 4:
                        // Tensor with NaN values
                        edge_tensor = torch::full({2, 2}, std::numeric_limits<float>::quiet_NaN());
                        break;
                    case 5:
                        // Very large tensor dimensions (but small total size)
                        edge_tensor = torch::ones({1, 1, 1, 1, 1, 1, 1, 1});
                        break;
                    case 6:
                        // Mixed special values
                        edge_tensor = torch::tensor({0.0f, 1.0f, -1.0f, 
                                                    std::numeric_limits<float>::infinity(),
                                                    -std::numeric_limits<float>::infinity(),
                                                    std::numeric_limits<float>::quiet_NaN()});
                        break;
                    case 7:
                        // Very small/large normal values
                        edge_tensor = torch::tensor({std::numeric_limits<float>::min(),
                                                    std::numeric_limits<float>::max(),
                                                    std::numeric_limits<float>::epsilon(),
                                                    -std::numeric_limits<float>::epsilon()});
                        break;
                }
                
                edge_tensor.erfc_();
                
            } catch (...) {
                // Edge cases might fail, that's ok
            }
        }
        
        // Test with complex numbers if supported
        if (offset + 4 < Size) {
            uint8_t complex_flag = Data[offset++];
            if (complex_flag & 0x01) {
                try {
                    auto complex_tensor = torch::randn({2, 2}, torch::kComplexFloat);
                    complex_tensor.erfc_();
                } catch (...) {
                    // Complex erfc might not be supported
                }
            }
        }
        
    }
    catch (const std::bad_alloc& e)
    {
        // Memory allocation failure - likely due to huge tensor dimensions
        // This is expected behavior, not a bug
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    catch (...)
    {
        // Unknown exception - this might be interesting
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }
    
    return 0; // keep the input
}