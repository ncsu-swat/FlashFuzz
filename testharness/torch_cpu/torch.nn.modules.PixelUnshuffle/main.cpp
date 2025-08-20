#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
            // But don't filter out 0 or negative values completely as they should trigger exceptions
            upscale_factor = std::abs(raw_factor) % 8 + 1;
            
            // Occasionally test with invalid values to check error handling
            if (raw_factor % 17 == 0) {
                upscale_factor = 0; // Invalid value
            } else if (raw_factor % 19 == 0) {
                upscale_factor = -1; // Invalid negative value
            }
        }
        
        // Create PixelUnshuffle module
        torch::nn::PixelUnshuffle pixelUnshuffle(upscale_factor);
        
        // Apply the operation
        torch::Tensor output = pixelUnshuffle->forward(input);
        
        // Optionally perform some validation on the output
        if (!output.defined()) {
            throw std::runtime_error("Output tensor is undefined");
        }
        
        // Test some edge cases with specific tensor shapes
        if (offset + 1 <= Size) {
            uint8_t edge_case_selector = Data[offset++];
            
            // Create tensors with specific shapes for edge cases
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
                    try {
                        torch::Tensor edge_output = pixelUnshuffle->forward(edge_input);
                    } catch (const std::exception& e) {
                        // Expected exception for incompatible dimensions
                    }
                }
            } else if (edge_case_selector % 5 == 2) {
                // Test with a 2D tensor (missing batch and channel dims)
                if (upscale_factor > 1) {
                    std::vector<int64_t> shape = {upscale_factor * 2, upscale_factor * 2};
                    torch::Tensor edge_input = torch::ones(shape);
                    try {
                        torch::Tensor edge_output = pixelUnshuffle->forward(edge_input);
                    } catch (const std::exception& e) {
                        // Expected exception for wrong number of dimensions
                    }
                }
            } else if (edge_case_selector % 5 == 3) {
                // Test with a 5D tensor (extra dimension)
                std::vector<int64_t> shape = {1, 1, upscale_factor, upscale_factor, 1};
                torch::Tensor edge_input = torch::ones(shape);
                try {
                    torch::Tensor edge_output = pixelUnshuffle->forward(edge_input);
                } catch (const std::exception& e) {
                    // Expected exception for wrong number of dimensions
                }
            } else {
                // Test with empty tensor (zero in one dimension)
                std::vector<int64_t> shape = {0, 1, upscale_factor, upscale_factor};
                torch::Tensor edge_input = torch::ones(shape);
                try {
                    torch::Tensor edge_output = pixelUnshuffle->forward(edge_input);
                } catch (const std::exception& e) {
                    // May or may not throw depending on implementation
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}