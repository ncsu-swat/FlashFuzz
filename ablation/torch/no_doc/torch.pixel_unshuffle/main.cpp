#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 3 bytes: 2 for tensor metadata + 1 for downscale_factor
        if (Size < 3) {
            return 0;
        }

        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse downscale_factor from remaining data
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t factor_byte = Data[offset++];
        // Limit downscale_factor to reasonable range [1, 16] to avoid memory issues
        int64_t downscale_factor = (factor_byte % 16) + 1;
        
        // Try different tensor configurations
        try {
            // Direct call with original tensor
            torch::Tensor result = torch::pixel_unshuffle(input, downscale_factor);
            
            // Verify output properties
            if (result.defined() && result.numel() > 0) {
                // Force computation
                result.sum().item<float>();
            }
        } catch (const c10::Error& e) {
            // Expected for invalid inputs (wrong dimensions, etc.)
        } catch (const std::exception& e) {
            // Other exceptions from tensor operations
        }
        
        // Try with reshaped tensors to create valid 4D inputs
        if (input.defined() && input.numel() > 0) {
            try {
                // Reshape to 4D if possible
                int64_t total_elements = input.numel();
                
                // Try different 4D shape configurations
                std::vector<std::vector<int64_t>> shapes_to_try;
                
                // Configuration 1: Balanced dimensions
                if (total_elements >= 4) {
                    int64_t dim = std::max(int64_t(1), int64_t(std::pow(total_elements, 0.25)));
                    if (dim * dim * dim * dim <= total_elements) {
                        shapes_to_try.push_back({1, dim, dim * dim, total_elements / (dim * dim * dim)});
                    }
                }
                
                // Configuration 2: Channel-heavy
                shapes_to_try.push_back({1, total_elements, 1, 1});
                
                // Configuration 3: Spatial-heavy  
                if (total_elements >= downscale_factor * downscale_factor) {
                    int64_t spatial_dim = downscale_factor * 2;
                    if (spatial_dim * spatial_dim <= total_elements) {
                        shapes_to_try.push_back({1, 1, spatial_dim, total_elements / spatial_dim});
                    }
                }
                
                // Configuration 4: Batch-heavy
                shapes_to_try.push_back({total_elements, 1, 1, 1});
                
                for (const auto& shape : shapes_to_try) {
                    try {
                        // Check if reshape is valid
                        int64_t shape_elements = 1;
                        for (auto dim : shape) {
                            if (dim <= 0) break;
                            shape_elements *= dim;
                        }
                        
                        if (shape_elements == total_elements) {
                            torch::Tensor reshaped = input.reshape(shape);
                            
                            // Ensure height and width are divisible by downscale_factor
                            if (shape[2] % downscale_factor == 0 && shape[3] % downscale_factor == 0) {
                                torch::Tensor result = torch::pixel_unshuffle(reshaped, downscale_factor);
                                
                                // Verify result properties
                                if (result.defined()) {
                                    // Check expected output shape
                                    int64_t expected_channels = shape[1] * downscale_factor * downscale_factor;
                                    int64_t expected_height = shape[2] / downscale_factor;
                                    int64_t expected_width = shape[3] / downscale_factor;
                                    
                                    if (result.size(0) == shape[0] && 
                                        result.size(1) == expected_channels &&
                                        result.size(2) == expected_height &&
                                        result.size(3) == expected_width) {
                                        // Shape is correct
                                        result.sum().item<float>();
                                    }
                                }
                            }
                        }
                    } catch (const c10::Error& e) {
                        // Continue with next configuration
                    } catch (const std::exception& e) {
                        // Continue with next configuration  
                    }
                }
            } catch (const std::exception& e) {
                // Reshape failed, continue
            }
            
            // Try with different memory layouts
            try {
                if (input.dim() == 4) {
                    // Make contiguous
                    torch::Tensor contiguous = input.contiguous();
                    if (contiguous.size(2) % downscale_factor == 0 && 
                        contiguous.size(3) % downscale_factor == 0) {
                        torch::pixel_unshuffle(contiguous, downscale_factor);
                    }
                    
                    // Try with permuted dimensions
                    torch::Tensor permuted = input.permute({0, 1, 3, 2});
                    if (permuted.size(2) % downscale_factor == 0 && 
                        permuted.size(3) % downscale_factor == 0) {
                        torch::pixel_unshuffle(permuted, downscale_factor);
                    }
                }
            } catch (const c10::Error& e) {
                // Expected for invalid operations
            } catch (const std::exception& e) {
                // Other exceptions
            }
            
            // Try with different dtypes
            try {
                std::vector<torch::ScalarType> dtypes_to_try = {
                    torch::kFloat32, torch::kFloat64, torch::kFloat16,
                    torch::kInt32, torch::kInt64, torch::kInt8
                };
                
                for (auto dtype : dtypes_to_try) {
                    try {
                        torch::Tensor converted = input.to(dtype);
                        if (converted.dim() == 4 && 
                            converted.size(2) % downscale_factor == 0 &&
                            converted.size(3) % downscale_factor == 0) {
                            torch::pixel_unshuffle(converted, downscale_factor);
                        }
                    } catch (const c10::Error& e) {
                        // Continue with next dtype
                    }
                }
            } catch (const std::exception& e) {
                // Type conversion failed
            }
            
            // Edge cases with extreme downscale factors
            if (offset < Size) {
                try {
                    // Try with different downscale factors
                    std::vector<int64_t> factors = {1, 2, 3, 4, 8, 
                                                    downscale_factor * 2,
                                                    downscale_factor / 2};
                    
                    for (auto factor : factors) {
                        if (factor > 0 && factor <= 32) {
                            try {
                                if (input.dim() == 4 &&
                                    input.size(2) % factor == 0 &&
                                    input.size(3) % factor == 0) {
                                    torch::pixel_unshuffle(input, factor);
                                }
                            } catch (const c10::Error& e) {
                                // Continue
                            }
                        }
                    }
                } catch (const std::exception& e) {
                    // Continue
                }
            }
            
            // Test with requires_grad
            try {
                if (input.dtype() == torch::kFloat32 || input.dtype() == torch::kFloat64) {
                    torch::Tensor grad_input = input.requires_grad_(true);
                    if (grad_input.dim() == 4 &&
                        grad_input.size(2) % downscale_factor == 0 &&
                        grad_input.size(3) % downscale_factor == 0) {
                        torch::Tensor result = torch::pixel_unshuffle(grad_input, downscale_factor);
                        if (result.defined() && result.requires_grad()) {
                            // Test backward pass
                            result.sum().backward();
                        }
                    }
                }
            } catch (const c10::Error& e) {
                // Gradient computation failed
            } catch (const std::exception& e) {
                // Other exceptions
            }
        }
        
        // Test with zero-element tensors
        try {
            torch::Tensor zero_tensor = torch::zeros({1, 1, 0, 0});
            torch::pixel_unshuffle(zero_tensor, 1);
            
            torch::Tensor zero_batch = torch::zeros({0, 1, 2, 2});
            torch::pixel_unshuffle(zero_batch, 1);
        } catch (const c10::Error& e) {
            // Expected for some zero configurations
        } catch (const std::exception& e) {
            // Other exceptions
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}