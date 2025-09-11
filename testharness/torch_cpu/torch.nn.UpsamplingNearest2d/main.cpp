#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 4 dimensions (N, C, H, W) for UpsamplingNearest2d
        if (input.dim() < 4) {
            // Add dimensions if needed
            while (input.dim() < 4) {
                input = input.unsqueeze(0);
            }
        }
        
        // Extract parameters for UpsamplingNearest2d
        // We need at least 4 bytes for scale_factor or output_size
        if (offset + 4 > Size) {
            return 0;
        }
        
        // Decide whether to use scale_factor or output_size
        bool use_scale_factor = (Data[offset++] % 2 == 0);
        
        if (use_scale_factor) {
            // Parse scale_factor (float value)
            float scale_factor = 1.0f;
            if (offset + sizeof(float) <= Size) {
                std::memcpy(&scale_factor, Data + offset, sizeof(float));
                offset += sizeof(float);
                scale_factor = std::abs(scale_factor);
                if (scale_factor == 0.0f) scale_factor = 1.0f;
            }
            
            // Apply upsampling using functional API
            torch::Tensor output = torch::nn::functional::upsample_nearest2d(
                input, 
                torch::nn::functional::UpsampleNearest2dFuncOptions().scale_factor(std::vector<double>{scale_factor, scale_factor})
            );
        } 
        else {
            // Parse output_size (two int values for height and width)
            int64_t output_height = 1;
            int64_t output_width = 1;
            
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&output_height, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Make output_height positive
                output_height = std::abs(output_height) % 100 + 1;
            }
            
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&output_width, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Make output_width positive
                output_width = std::abs(output_width) % 100 + 1;
            }
            
            // Apply upsampling using functional API
            torch::Tensor output = torch::nn::functional::upsample_nearest2d(
                input,
                torch::nn::functional::UpsampleNearest2dFuncOptions().size(std::vector<int64_t>{output_height, output_width})
            );
        }
        
        // Try alternative forms
        if (offset + 1 < Size) {
            uint8_t alt_mode = Data[offset++];
            
            if (alt_mode % 3 == 0) {
                // Test with just a scale factor value
                float scale = 1.5f;
                if (offset + sizeof(float) <= Size) {
                    std::memcpy(&scale, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    scale = std::abs(scale);
                    if (scale == 0.0f) scale = 1.0f;
                }
                
                torch::Tensor output = torch::nn::functional::upsample_nearest2d(
                    input,
                    torch::nn::functional::UpsampleNearest2dFuncOptions().scale_factor(std::vector<double>{scale, scale})
                );
            }
            else if (alt_mode % 3 == 1) {
                // Test with a vector of scale factors
                std::vector<double> scales;
                float scale1 = 1.0f, scale2 = 1.0f;
                
                if (offset + sizeof(float) <= Size) {
                    std::memcpy(&scale1, Data + offset, sizeof(float));
                    offset += sizeof(float);
                }
                
                if (offset + sizeof(float) <= Size) {
                    std::memcpy(&scale2, Data + offset, sizeof(float));
                    offset += sizeof(float);
                }
                
                scales.push_back(std::abs(scale1) + 0.1);
                scales.push_back(std::abs(scale2) + 0.1);
                
                torch::Tensor output = torch::nn::functional::upsample_nearest2d(
                    input,
                    torch::nn::functional::UpsampleNearest2dFuncOptions().scale_factor(scales)
                );
            }
            else {
                // Test with a vector for output size
                std::vector<int64_t> sizes;
                int64_t size1 = 1, size2 = 1;
                
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&size1, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                }
                
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&size2, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                }
                
                sizes.push_back(std::abs(size1) % 100 + 1);
                sizes.push_back(std::abs(size2) % 100 + 1);
                
                torch::Tensor output = torch::nn::functional::upsample_nearest2d(
                    input,
                    torch::nn::functional::UpsampleNearest2dFuncOptions().size(sizes)
                );
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
