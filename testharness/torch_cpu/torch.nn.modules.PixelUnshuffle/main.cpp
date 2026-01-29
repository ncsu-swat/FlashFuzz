#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Need at least a few bytes for meaningful fuzzing
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract upscale_factor from the remaining data
        int64_t upscale_factor = 2; // Default value
        if (offset + sizeof(int64_t) <= Size) {
            int64_t raw_factor;
            std::memcpy(&raw_factor, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure upscale_factor is positive (required by PixelUnshuffle)
            upscale_factor = std::abs(raw_factor) % 8 + 1;
            
            // Occasionally test with invalid values to check error handling
            if (raw_factor % 17 == 0) {
                upscale_factor = 0; // Invalid value
            } else if (raw_factor % 19 == 0) {
                upscale_factor = -1; // Invalid negative value
            }
        }
        
        // Create PixelUnshuffle module - may throw for invalid upscale_factor
        torch::nn::PixelUnshuffle pixelUnshuffle(upscale_factor);
        
        // Apply the operation - may throw for incompatible input shapes
        try {
            torch::Tensor output = pixelUnshuffle->forward(input);
            
            // Optionally perform some validation on the output
            if (output.defined()) {
                // Access some properties to ensure the tensor is valid
                auto sizes = output.sizes();
                auto numel = output.numel();
                (void)sizes;
                (void)numel;
            }
        } catch (const std::exception& e) {
            // Expected for random tensor shapes that don't meet PixelUnshuffle requirements
        }
        
        // Test some edge cases with specific tensor shapes
        if (offset + 1 <= Size) {
            uint8_t edge_case_selector = Data[offset++];
            
            // Only test edge cases with valid upscale_factor
            if (upscale_factor > 0) {
                try {
                    if (edge_case_selector % 5 == 0) {
                        // Test with a tensor that has exactly the right dimensions for unshuffle
                        std::vector<int64_t> shape = {1, 1, upscale_factor, upscale_factor};
                        torch::Tensor edge_input = torch::ones(shape);
                        torch::Tensor edge_output = pixelUnshuffle->forward(edge_input);
                    } else if (edge_case_selector % 5 == 1) {
                        // Test with a tensor that has dimensions not divisible by upscale_factor
                        if (upscale_factor > 1) {
                            std::vector<int64_t> shape = {1, 3, upscale_factor + 1, upscale_factor + 1};
                            torch::Tensor edge_input = torch::ones(shape);
                            torch::Tensor edge_output = pixelUnshuffle->forward(edge_input);
                        }
                    } else if (edge_case_selector % 5 == 2) {
                        // Test with a 2D tensor (missing batch and channel dims)
                        std::vector<int64_t> shape = {upscale_factor * 2, upscale_factor * 2};
                        torch::Tensor edge_input = torch::ones(shape);
                        torch::Tensor edge_output = pixelUnshuffle->forward(edge_input);
                    } else if (edge_case_selector % 5 == 3) {
                        // Test with a 5D tensor (extra dimension)
                        std::vector<int64_t> shape = {1, 1, upscale_factor, upscale_factor, 1};
                        torch::Tensor edge_input = torch::ones(shape);
                        torch::Tensor edge_output = pixelUnshuffle->forward(edge_input);
                    } else {
                        // Test with larger valid dimensions
                        std::vector<int64_t> shape = {2, 4, upscale_factor * 3, upscale_factor * 3};
                        torch::Tensor edge_input = torch::randn(shape);
                        torch::Tensor edge_output = pixelUnshuffle->forward(edge_input);
                    }
                } catch (const std::exception& e) {
                    // Expected exceptions for incompatible shapes/dimensions
                }
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