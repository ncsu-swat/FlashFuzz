#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal data to create a tensor
        if (Size < 2) {
            return 0;  // Not enough data, but keep for coverage
        }

        // Create the input tensor from fuzzer data
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If we can't create a basic tensor, try with minimal valid tensor
            if (Size >= 1) {
                // Create a scalar tensor with the available dtype selector
                auto dtype = fuzzer_utils::parseDataType(Data[0]);
                auto options = torch::TensorOptions().dtype(dtype);
                input_tensor = torch::zeros({}, options);  // scalar tensor
            } else {
                return 0;  // Keep input for coverage
            }
        }

        // Test various device configurations if there's extra data
        torch::Device device = torch::kCPU;
        if (offset < Size) {
            uint8_t device_selector = Data[offset++];
            if (torch::cuda::is_available() && (device_selector % 4 == 0)) {
                device = torch::kCUDA;
                try {
                    input_tensor = input_tensor.to(device);
                } catch (...) {
                    // CUDA operation failed, continue with CPU
                    device = torch::kCPU;
                }
            }
        }

        // Test various memory layouts if there's extra data
        if (offset < Size) {
            uint8_t layout_selector = Data[offset++];
            try {
                switch (layout_selector % 5) {
                    case 0:
                        // Keep original layout
                        break;
                    case 1:
                        // Make non-contiguous via transpose if possible
                        if (input_tensor.dim() >= 2) {
                            input_tensor = input_tensor.transpose(0, 1);
                        }
                        break;
                    case 2:
                        // Make contiguous
                        input_tensor = input_tensor.contiguous();
                        break;
                    case 3:
                        // Try to create a view if tensor has elements
                        if (input_tensor.numel() > 1) {
                            auto flat = input_tensor.flatten();
                            if (flat.numel() > 0) {
                                input_tensor = flat.view({-1});
                            }
                        }
                        break;
                    case 4:
                        // Try squeeze/unsqueeze operations
                        if (input_tensor.dim() > 0) {
                            input_tensor = input_tensor.unsqueeze(0).squeeze(0);
                        }
                        break;
                }
            } catch (...) {
                // Layout manipulation failed, continue with current tensor
            }
        }

        // Main operation: torch.rand_like
        torch::Tensor result;
        try {
            result = torch::rand_like(input_tensor);
            
            // Verify basic properties
            if (result.sizes() != input_tensor.sizes()) {
                std::cerr << "Shape mismatch: expected " << input_tensor.sizes() 
                         << " got " << result.sizes() << std::endl;
            }
            
            if (result.dtype() != input_tensor.dtype()) {
                std::cerr << "Dtype mismatch: expected " << input_tensor.dtype() 
                         << " got " << result.dtype() << std::endl;
            }
            
            if (result.device() != input_tensor.device()) {
                std::cerr << "Device mismatch: expected " << input_tensor.device() 
                         << " got " << result.device() << std::endl;
            }

            // For floating point types, verify values are in [0, 1)
            if (result.is_floating_point() && result.numel() > 0) {
                auto min_val = result.min();
                auto max_val = result.max();
                
                // Move to CPU for item() if needed
                if (min_val.device().type() != torch::kCPU) {
                    min_val = min_val.cpu();
                    max_val = max_val.cpu();
                }
                
                double min_v = min_val.item<double>();
                double max_v = max_val.item<double>();
                
                if (min_v < 0.0 || max_v >= 1.0) {
                    std::cerr << "Values out of range [0, 1): min=" << min_v 
                             << " max=" << max_v << std::endl;
                }
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors
            // Some dtypes might not support rand_like, continue fuzzing
            return 0;
        }

        // Test with additional parameters if more data available
        if (offset + 1 < Size) {
            uint8_t param_selector = Data[offset++];
            
            // Test with explicit dtype override
            if (param_selector % 3 == 0 && offset < Size) {
                auto new_dtype = fuzzer_utils::parseDataType(Data[offset++]);
                try {
                    auto options = torch::TensorOptions()
                        .dtype(new_dtype)
                        .device(device);
                    result = torch::rand_like(input_tensor, options);
                } catch (...) {
                    // Some dtype combinations might be invalid
                }
            }
            
            // Test with memory format hints
            if (param_selector % 3 == 1) {
                try {
                    auto memory_format = (param_selector % 2 == 0) ? 
                        torch::MemoryFormat::Contiguous : 
                        torch::MemoryFormat::Preserve;
                    result = torch::rand_like(input_tensor, input_tensor.options(), memory_format);
                } catch (...) {
                    // Memory format might not be applicable
                }
            }
            
            // Test with layout changes
            if (param_selector % 3 == 2) {
                try {
                    auto options = input_tensor.options();
                    // Try sparse tensors for specific shapes
                    if (input_tensor.dim() == 2 && param_selector % 5 == 0) {
                        options = options.layout(torch::kSparse);
                    }
                    result = torch::rand_like(input_tensor, options);
                } catch (...) {
                    // Layout change might not be supported
                }
            }
        }

        // Test edge cases with zero-element tensors
        if (input_tensor.numel() == 0) {
            try {
                auto zero_result = torch::rand_like(input_tensor);
                if (zero_result.numel() != 0) {
                    std::cerr << "Zero-element tensor produced non-zero result" << std::endl;
                }
            } catch (...) {
                // Zero-element handling might vary
            }
        }

        // Test with requires_grad if applicable
        if (offset < Size && input_tensor.is_floating_point()) {
            uint8_t grad_selector = Data[offset++];
            if (grad_selector % 2 == 0) {
                try {
                    input_tensor = input_tensor.requires_grad_(true);
                    result = torch::rand_like(input_tensor);
                    // Result should not require grad by default
                    if (result.requires_grad()) {
                        std::cerr << "Unexpected requires_grad propagation" << std::endl;
                    }
                } catch (...) {
                    // Gradient operations might fail
                }
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    catch (...)
    {
        std::cout << "Exception caught: Unknown exception" << std::endl;
        return -1; // discard the input  
    }
    
    return 0; // keep the input
}