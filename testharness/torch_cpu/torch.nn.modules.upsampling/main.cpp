#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for upsampling
        bool align_corners = false;
        bool scale_factor_specified = false;
        bool size_specified = false;
        
        if (offset < Size) {
            align_corners = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            scale_factor_specified = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            size_specified = Data[offset++] & 0x1;
        }
        
        // Get input dimensions
        int64_t dim = input.dim();
        
        // Try different upsampling modes
        std::vector<torch::nn::UpsampleOptions::mode_t> modes = {
            torch::kNearest, torch::kLinear, torch::kBilinear, torch::kBicubic, torch::kTrilinear
        };
        torch::nn::UpsampleOptions::mode_t mode = torch::kNearest;
        
        if (offset < Size) {
            mode = modes[Data[offset++] % modes.size()];
        }
        
        // Create upsampling module and apply it
        if (dim >= 3) {
            if (scale_factor_specified) {
                // Create scale factors
                std::vector<double> scale_factors;
                for (int i = 2; i < dim; i++) {
                    double scale = 1.0;
                    if (offset + sizeof(double) <= Size) {
                        memcpy(&scale, Data + offset, sizeof(double));
                        offset += sizeof(double);
                    }
                    scale_factors.push_back(scale);
                }
                
                // Create Upsample module with scale_factor
                torch::nn::Upsample upsample = torch::nn::Upsample(
                    torch::nn::UpsampleOptions()
                        .mode(mode)
                        .align_corners(align_corners)
                        .scale_factor(scale_factors)
                );
                
                // Apply upsampling
                torch::Tensor output = upsample->forward(input);
            } else if (size_specified) {
                // Create size vector
                std::vector<int64_t> sizes;
                for (int i = 2; i < dim; i++) {
                    int64_t size_val = 1;
                    if (offset + sizeof(int64_t) <= Size) {
                        memcpy(&size_val, Data + offset, sizeof(int64_t));
                        offset += sizeof(int64_t);
                        // Ensure size is positive
                        size_val = std::abs(size_val) % 100 + 1;
                    }
                    sizes.push_back(size_val);
                }
                
                // Create Upsample module with size
                torch::nn::Upsample upsample = torch::nn::Upsample(
                    torch::nn::UpsampleOptions()
                        .mode(mode)
                        .align_corners(align_corners)
                        .size(sizes)
                );
                
                // Apply upsampling
                torch::Tensor output = upsample->forward(input);
            } else {
                // Default case: use a simple scale factor of 2
                torch::nn::Upsample upsample = torch::nn::Upsample(
                    torch::nn::UpsampleOptions()
                        .mode(mode)
                        .align_corners(align_corners)
                        .scale_factor(2.0)
                );
                
                // Apply upsampling
                torch::Tensor output = upsample->forward(input);
            }
        }
        
        // Test general upsampling with different configurations
        if (dim >= 3) {
            double scale = 1.5;
            if (offset + sizeof(double) <= Size) {
                memcpy(&scale, Data + offset, sizeof(double));
                offset += sizeof(double);
            }
            
            // Test with scale factor
            torch::nn::Upsample upsample_scale = torch::nn::Upsample(
                torch::nn::UpsampleOptions()
                    .mode(torch::kNearest)
                    .scale_factor(scale)
            );
            torch::Tensor output_scale = upsample_scale->forward(input);
            
            // Test with size specification
            std::vector<int64_t> target_sizes;
            for (int i = 2; i < dim; i++) {
                target_sizes.push_back(input.size(i) * 2);
            }
            
            torch::nn::Upsample upsample_size = torch::nn::Upsample(
                torch::nn::UpsampleOptions()
                    .mode(torch::kNearest)
                    .size(target_sizes)
            );
            torch::Tensor output_size = upsample_size->forward(input);
            
            // Test bilinear mode for 4D tensors
            if (dim == 4) {
                torch::nn::Upsample upsample_bilinear = torch::nn::Upsample(
                    torch::nn::UpsampleOptions()
                        .mode(torch::kBilinear)
                        .scale_factor(scale)
                        .align_corners(align_corners)
                );
                torch::Tensor output_bilinear = upsample_bilinear->forward(input);
            }
            
            // Test trilinear mode for 5D tensors
            if (dim == 5) {
                torch::nn::Upsample upsample_trilinear = torch::nn::Upsample(
                    torch::nn::UpsampleOptions()
                        .mode(torch::kTrilinear)
                        .scale_factor(scale)
                        .align_corners(align_corners)
                );
                torch::Tensor output_trilinear = upsample_trilinear->forward(input);
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