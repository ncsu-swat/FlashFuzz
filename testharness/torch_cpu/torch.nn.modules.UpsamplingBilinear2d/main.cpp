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
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // UpsamplingBilinear2d requires floating point input
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Ensure input has 4 dimensions (N x C x H x W) for UpsamplingBilinear2d
        while (input.dim() < 4) {
            input = input.unsqueeze(0);
        }
        // If more than 4 dims, flatten extra dims into batch
        while (input.dim() > 4) {
            input = input.squeeze(0);
        }
        
        // Ensure spatial dimensions are at least 1
        if (input.size(2) < 1 || input.size(3) < 1) {
            return 0;
        }
        
        // Extract parameters for UpsamplingBilinear2d from the remaining data
        uint32_t size_param = 0;
        if (offset + 4 <= Size) {
            std::memcpy(&size_param, Data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
        } else if (offset < Size) {
            size_param = Data[offset];
            offset++;
        }
        
        // Get output size parameters - ensure they are reasonable
        int64_t output_h = (size_param % 64) + 1; // Range 1-64
        int64_t output_w = ((size_param >> 8) % 64) + 1; // Range 1-64
        
        // Determine align_corners setting
        bool align_corners = (size_param & 0x2000) != 0;
        
        // Inner try-catch for expected operational failures
        try {
            torch::nn::Upsample upsampler = nullptr;
            
            // Choose between different constructor forms based on fuzzer data
            int mode_choice = size_param % 3;
            
            if (mode_choice == 0) {
                // Use a single scale factor
                double scale_factor = ((size_param % 20) + 5) / 10.0; // Range 0.5 to 2.4
                upsampler = torch::nn::Upsample(
                    torch::nn::UpsampleOptions()
                        .scale_factor(std::vector<double>{scale_factor, scale_factor})
                        .mode(torch::kBilinear)
                        .align_corners(align_corners));
            } else if (mode_choice == 1) {
                // Use different scale factors for height and width
                double scale_h = ((size_param % 20) + 5) / 10.0; // Range 0.5 to 2.4
                double scale_w = (((size_param >> 4) % 20) + 5) / 10.0; // Range 0.5 to 2.4
                upsampler = torch::nn::Upsample(
                    torch::nn::UpsampleOptions()
                        .scale_factor(std::vector<double>{scale_h, scale_w})
                        .mode(torch::kBilinear)
                        .align_corners(align_corners));
            } else {
                // Use explicit output size
                upsampler = torch::nn::Upsample(
                    torch::nn::UpsampleOptions()
                        .size(std::vector<int64_t>{output_h, output_w})
                        .mode(torch::kBilinear)
                        .align_corners(align_corners));
            }
            
            // Apply the upsampling operation
            torch::Tensor output = upsampler->forward(input);
            
            // Verify output has expected properties
            (void)output.sizes();
            (void)output.numel();
        }
        catch (const c10::Error&) {
            // Expected failures (shape mismatches, etc.) - silently ignore
        }
        catch (const std::runtime_error&) {
            // Expected runtime errors - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}