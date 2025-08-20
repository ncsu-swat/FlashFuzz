#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (N x C x H x W) for UpsamplingBilinear2d
        if (input.dim() < 3) {
            // Unsqueeze to add dimensions if needed
            while (input.dim() < 3) {
                input = input.unsqueeze(0);
            }
            // Add one more dimension if needed to make it 4D
            if (input.dim() == 3) {
                input = input.unsqueeze(0);
            }
        }
        
        // Extract parameters for UpsamplingBilinear2d from the remaining data
        if (offset + 4 <= Size) {
            // Get output size parameters
            uint32_t size_param;
            std::memcpy(&size_param, Data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            
            // Determine output size - either a single value or a pair
            int64_t output_h = (size_param % 64) + 1; // Ensure positive
            int64_t output_w = ((size_param >> 8) % 64) + 1; // Ensure positive
            
            // Create UpsamplingBilinear2d module with output size
            torch::nn::Upsample upsampler = nullptr;
            
            // Randomly choose between different constructor forms
            if (size_param & 0x1000) {
                // Use a single scale factor
                double scale_factor = (size_param % 10) / 10.0 + 0.5; // Range 0.5 to 1.4
                upsampler = torch::nn::Upsample(
                    torch::nn::UpsampleOptions().scale_factor({scale_factor, scale_factor}).mode(torch::kBilinear).align_corners(size_param & 0x2000));
            } else if (size_param & 0x4000) {
                // Use different scale factors for height and width
                double scale_h = (size_param % 10) / 10.0 + 0.5; // Range 0.5 to 1.4
                double scale_w = ((size_param >> 4) % 10) / 10.0 + 0.5; // Range 0.5 to 1.4
                upsampler = torch::nn::Upsample(
                    torch::nn::UpsampleOptions().scale_factor({scale_h, scale_w}).mode(torch::kBilinear).align_corners(size_param & 0x2000));
            } else {
                // Use output size
                upsampler = torch::nn::Upsample(
                    torch::nn::UpsampleOptions().size({output_h, output_w}).mode(torch::kBilinear).align_corners(size_param & 0x2000));
            }
            
            // Apply the upsampling operation
            torch::Tensor output = upsampler->forward(input);
        } else {
            // Not enough data for parameters, use default constructor
            torch::nn::Upsample upsampler = torch::nn::Upsample(
                torch::nn::UpsampleOptions().size({2, 2}).mode(torch::kBilinear));
            
            // Apply the upsampling operation
            torch::Tensor output = upsampler->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}