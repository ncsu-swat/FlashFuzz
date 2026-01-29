#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 4 dimensions (N, C, H, W) for UpsamplingNearest2d
        if (input.dim() < 4) {
            while (input.dim() < 4) {
                input = input.unsqueeze(0);
            }
        }
        
        // Limit input spatial dimensions to avoid OOM
        if (input.size(2) > 64 || input.size(3) > 64) {
            input = input.slice(2, 0, std::min(input.size(2), (int64_t)64));
            input = input.slice(3, 0, std::min(input.size(3), (int64_t)64));
        }
        
        // Ensure float type for interpolation
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Need parameters
        if (offset + 4 > Size) {
            return 0;
        }
        
        // Decide whether to use scale_factor or output_size
        bool use_scale_factor = (Data[offset++] % 2 == 0);
        
        // Helper lambda to sanitize scale factor
        auto sanitize_scale = [](float val) -> double {
            if (std::isnan(val) || std::isinf(val) || val <= 0.0f) {
                return 1.0;
            }
            // Limit scale factor to avoid OOM (max 10x upscale)
            return std::min(std::max(static_cast<double>(val), 0.1), 10.0);
        };
        
        volatile float result_sum = 0.0f;  // Prevent optimization
        
        if (use_scale_factor) {
            float scale_factor = 1.0f;
            if (offset + sizeof(float) <= Size) {
                std::memcpy(&scale_factor, Data + offset, sizeof(float));
                offset += sizeof(float);
            }
            
            double safe_scale = sanitize_scale(scale_factor);
            
            try {
                // Use torch::nn::Upsample module with nearest mode
                auto upsample = torch::nn::Upsample(
                    torch::nn::UpsampleOptions()
                        .scale_factor(std::vector<double>{safe_scale, safe_scale})
                        .mode(torch::kNearest)
                );
                torch::Tensor output = upsample->forward(input);
                result_sum += output.numel();
            } catch (const c10::Error&) {
                // Expected for invalid configurations
            }
        } 
        else {
            int64_t output_height = 1;
            int64_t output_width = 1;
            
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&output_height, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                output_height = std::abs(output_height) % 256 + 1;
            }
            
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&output_width, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                output_width = std::abs(output_width) % 256 + 1;
            }
            
            try {
                auto upsample = torch::nn::Upsample(
                    torch::nn::UpsampleOptions()
                        .size(std::vector<int64_t>{output_height, output_width})
                        .mode(torch::kNearest)
                );
                torch::Tensor output = upsample->forward(input);
                result_sum += output.numel();
            } catch (const c10::Error&) {
                // Expected for invalid configurations
            }
        }
        
        // Try alternative forms - using functional::interpolate
        if (offset + 1 < Size) {
            uint8_t alt_mode = Data[offset++];
            
            if (alt_mode % 3 == 0) {
                float scale = 1.5f;
                if (offset + sizeof(float) <= Size) {
                    std::memcpy(&scale, Data + offset, sizeof(float));
                    offset += sizeof(float);
                }
                
                double safe_scale = sanitize_scale(scale);
                
                try {
                    auto options = torch::nn::functional::InterpolateFuncOptions()
                        .scale_factor(std::vector<double>{safe_scale, safe_scale})
                        .mode(torch::kNearest);
                    torch::Tensor output = torch::nn::functional::interpolate(input, options);
                    result_sum += output.numel();
                } catch (const c10::Error&) {
                    // Expected for invalid configurations
                }
            }
            else if (alt_mode % 3 == 1) {
                float scale1 = 1.0f, scale2 = 1.0f;
                
                if (offset + sizeof(float) <= Size) {
                    std::memcpy(&scale1, Data + offset, sizeof(float));
                    offset += sizeof(float);
                }
                
                if (offset + sizeof(float) <= Size) {
                    std::memcpy(&scale2, Data + offset, sizeof(float));
                    offset += sizeof(float);
                }
                
                double safe_scale1 = sanitize_scale(scale1);
                double safe_scale2 = sanitize_scale(scale2);
                
                try {
                    auto options = torch::nn::functional::InterpolateFuncOptions()
                        .scale_factor(std::vector<double>{safe_scale1, safe_scale2})
                        .mode(torch::kNearest);
                    torch::Tensor output = torch::nn::functional::interpolate(input, options);
                    result_sum += output.numel();
                } catch (const c10::Error&) {
                    // Expected for invalid configurations
                }
            }
            else {
                int64_t size1 = 1, size2 = 1;
                
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&size1, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                }
                
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&size2, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                }
                
                size1 = std::abs(size1) % 256 + 1;
                size2 = std::abs(size2) % 256 + 1;
                
                try {
                    auto options = torch::nn::functional::InterpolateFuncOptions()
                        .size(std::vector<int64_t>{size1, size2})
                        .mode(torch::kNearest);
                    torch::Tensor output = torch::nn::functional::interpolate(input, options);
                    result_sum += output.numel();
                } catch (const c10::Error&) {
                    // Expected for invalid configurations
                }
            }
        }
        
        // Also test the UpsamplingNearest2d module directly (alias for Upsample with nearest mode)
        if (offset + 2 < Size) {
            int64_t h = (Data[offset++] % 64) + 1;
            int64_t w = (Data[offset++] % 64) + 1;
            
            try {
                // UpsamplingNearest2d is essentially Upsample with mode=nearest
                auto upsample_nearest = torch::nn::Upsample(
                    torch::nn::UpsampleOptions()
                        .size(std::vector<int64_t>{h, w})
                        .mode(torch::kNearest)
                );
                torch::Tensor output = upsample_nearest->forward(input);
                result_sum += output.numel();
            } catch (const c10::Error&) {
                // Expected for invalid configurations
            }
        }
        
        (void)result_sum;  // Suppress unused warning
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}