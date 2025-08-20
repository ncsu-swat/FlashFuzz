#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (N, C, H, W) for UpsamplingBilinear2d
        if (input.dim() < 3) {
            // Add dimensions if needed
            while (input.dim() < 3) {
                input = input.unsqueeze(0);
            }
            // Add one more dimension if needed to make it 4D (N, C, H, W)
            if (input.dim() == 3) {
                input = input.unsqueeze(0);
            }
        }
        
        // Extract parameters for UpsamplingBilinear2d
        // We need at least 4 bytes for size parameters
        if (offset + 4 <= Size) {
            // Extract output size or scale factor
            uint8_t use_size_flag = Data[offset++];
            bool use_size = (use_size_flag % 2 == 0);
            
            if (use_size) {
                // Use output_size parameter
                uint8_t h_size_byte = Data[offset++];
                uint8_t w_size_byte = Data[offset++];
                
                // Convert to reasonable output sizes
                int64_t output_h = (h_size_byte % 32) + 1; // 1-32
                int64_t output_w = (w_size_byte % 32) + 1; // 1-32
                
                // Create options with output_size
                auto options = torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{output_h, output_w})
                    .mode(torch::kBilinear);
                
                // Apply upsampling
                auto output = torch::nn::functional::interpolate(input, options);
            } else {
                // Use scale_factor parameter
                uint8_t scale_byte = Data[offset++];
                
                // Convert to reasonable scale factor (0.1 to 5.0)
                double scale_factor = (scale_byte % 50) / 10.0 + 0.1;
                
                // Create options with scale_factor
                auto options = torch::nn::functional::InterpolateFuncOptions()
                    .scale_factor(std::vector<double>{scale_factor, scale_factor})
                    .mode(torch::kBilinear);
                
                // Apply upsampling
                auto output = torch::nn::functional::interpolate(input, options);
            }
        } else {
            // Not enough data for parameters, use default
            auto options = torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{2, 2})
                .mode(torch::kBilinear);
            auto output = torch::nn::functional::interpolate(input, options);
        }
        
        // Try with align_corners option if we have more data
        if (offset + 1 <= Size) {
            uint8_t align_corners_byte = Data[offset++];
            bool align_corners = (align_corners_byte % 2 == 0);
            
            auto options = torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{2, 2})
                .mode(torch::kBilinear)
                .align_corners(align_corners);
            auto output = torch::nn::functional::interpolate(input, options);
        }
        
        // Try with both scale_factor and align_corners
        if (offset + 1 <= Size) {
            uint8_t scale_byte = Data[offset++];
            double scale_factor = (scale_byte % 50) / 10.0 + 0.1;
            
            bool align_corners = ((offset < Size) && (Data[offset++] % 2 == 0));
            
            auto options = torch::nn::functional::InterpolateFuncOptions()
                .scale_factor(std::vector<double>{scale_factor, scale_factor})
                .mode(torch::kBilinear)
                .align_corners(align_corners);
            
            auto output = torch::nn::functional::interpolate(input, options);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}