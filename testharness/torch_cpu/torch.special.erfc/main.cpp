#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for torch.special.erfc
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // torch.special.erfc requires floating point input
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Apply torch.special.erfc operation
        torch::Tensor result = torch::special::erfc(input);
        
        // Try some edge cases with modified tensors if we have enough data
        if (offset + 1 < Size) {
            torch::Tensor extreme_input;
            
            uint8_t selector = Data[offset++];
            if (selector % 4 == 0) {
                // Very large values
                extreme_input = input * 1e10;
            } else if (selector % 4 == 1) {
                // Very small values
                extreme_input = input * 1e-10;
            } else if (selector % 4 == 2) {
                // NaN values
                extreme_input = torch::full_like(input, std::numeric_limits<float>::quiet_NaN());
            } else {
                // Inf values
                extreme_input = torch::full_like(input, std::numeric_limits<float>::infinity());
            }
            
            // Apply torch.special.erfc to the extreme input
            try {
                torch::Tensor extreme_result = torch::special::erfc(extreme_input);
            } catch (...) {
                // Silently ignore expected failures with extreme values
            }
        }
        
        // Try with different tensor views if possible
        if (!input.sizes().empty() && input.numel() > 1) {
            // Create a view with different strides
            try {
                std::vector<int64_t> new_shape;
                int64_t total_elements = 1;
                
                for (int i = 0; i < input.dim(); i++) {
                    if (input.size(i) > 1) {
                        new_shape.push_back(input.size(i));
                        total_elements *= input.size(i);
                    }
                }
                
                if (!new_shape.empty() && total_elements == input.numel()) {
                    torch::Tensor reshaped = input.reshape(new_shape);
                    torch::Tensor result_reshaped = torch::special::erfc(reshaped);
                }
            } catch (...) {
                // Silently ignore reshape failures
            }
            
            // Try with transposed tensor if it has at least 2 dimensions
            if (input.dim() >= 2) {
                try {
                    torch::Tensor transposed = input.transpose(0, input.dim() - 1);
                    torch::Tensor result_transposed = torch::special::erfc(transposed);
                } catch (...) {
                    // Silently ignore transpose failures
                }
            }
        }
        
        // Test with contiguous tensor
        try {
            torch::Tensor contiguous_input = input.contiguous();
            torch::Tensor result_contiguous = torch::special::erfc(contiguous_input);
        } catch (...) {
            // Silently ignore failures
        }
        
        // Test with output tensor (out parameter variant)
        try {
            torch::Tensor out_tensor = torch::empty_like(input);
            torch::special::erfc_out(out_tensor, input);
        } catch (...) {
            // Silently ignore out variant failures
        }
        
        // Test with negative values
        try {
            torch::Tensor neg_input = -torch::abs(input);
            torch::Tensor result_neg = torch::special::erfc(neg_input);
        } catch (...) {
            // Silently ignore failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}