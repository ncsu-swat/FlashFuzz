#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>
#include <limits>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // erfcx requires floating point tensors
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Apply torch.special.erfcx operation
        torch::Tensor result = torch::special::erfcx(input);
        
        // Try some edge cases with modified tensors if we have enough data
        if (offset + 1 < Size) {
            torch::Tensor extreme_input;
            
            uint8_t selector = Data[offset++];
            if (selector % 4 == 0) {
                // Very large positive values
                extreme_input = input * 1e10;
            } else if (selector % 4 == 1) {
                // Very large negative values  
                extreme_input = input * -1e10;
            } else if (selector % 4 == 2) {
                // Values close to zero
                extreme_input = input * 1e-10;
            } else {
                // NaN and Inf values
                extreme_input = input.clone();
                if (extreme_input.numel() > 0) {
                    try {
                        auto flat = extreme_input.flatten();
                        if (flat.numel() > 0) {
                            flat.index_put_({0}, std::numeric_limits<float>::infinity());
                        }
                        if (flat.numel() > 1) {
                            flat.index_put_({1}, -std::numeric_limits<float>::infinity());
                        }
                        if (flat.numel() > 2) {
                            flat.index_put_({2}, std::nanf(""));
                        }
                    } catch (...) {
                        // Silently ignore indexing failures
                    }
                }
            }
            
            // Apply erfcx to the extreme input
            torch::Tensor extreme_result = torch::special::erfcx(extreme_input);
        }
        
        // Try with different tensor options if we have more data
        if (offset + 1 < Size) {
            uint8_t option_selector = Data[offset++];
            
            if (input.numel() > 0 && input.dim() > 0) {
                torch::Tensor modified_input;
                
                try {
                    if (option_selector % 3 == 0 && input.dim() > 1) {
                        // Transpose the tensor
                        modified_input = input.transpose(0, input.dim() - 1);
                    } else if (option_selector % 3 == 1) {
                        // Create a non-contiguous slice
                        modified_input = input;
                        for (int64_t d = 0; d < input.dim(); d++) {
                            if (input.size(d) > 2) {
                                modified_input = modified_input.slice(d, 0, input.size(d) - 1, 2);
                                break;
                            }
                        }
                    } else {
                        // Create a view with different shape
                        modified_input = input.reshape({-1});
                    }
                    
                    // Apply erfcx to the modified input
                    if (modified_input.defined() && modified_input.numel() > 0) {
                        torch::Tensor modified_result = torch::special::erfcx(modified_input);
                    }
                } catch (...) {
                    // Silently ignore tensor manipulation failures
                }
            }
        }
        
        // Test with different dtypes
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            try {
                torch::Tensor typed_input;
                if (dtype_selector % 2 == 0) {
                    typed_input = input.to(torch::kFloat64);
                } else {
                    typed_input = input.to(torch::kFloat32);
                }
                torch::Tensor typed_result = torch::special::erfcx(typed_input);
            } catch (...) {
                // Silently ignore dtype conversion failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}