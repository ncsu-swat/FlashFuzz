#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get input dimensions
        int64_t dim = input.dim();
        
        // Upsampling requires at least 3D tensor (batch, channels, spatial...)
        if (dim < 3 || dim > 5) {
            return 0;
        }
        
        // Extract parameters for upsampling
        bool align_corners = false;
        bool use_scale_factor = true;
        uint8_t mode_selector = 0;
        
        if (offset < Size) {
            align_corners = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            use_scale_factor = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            mode_selector = Data[offset++];
        }
        
        // Select appropriate mode based on tensor dimensions
        // We'll use an integer to track mode and set options accordingly
        int mode_type = 0; // 0=nearest, 1=linear, 2=bilinear, 3=bicubic, 4=trilinear
        bool mode_supports_align_corners = false;
        
        if (dim == 3) {
            // 3D: nearest or linear
            if (mode_selector & 0x1) {
                mode_type = 1; // linear
                mode_supports_align_corners = true;
            } else {
                mode_type = 0; // nearest
                mode_supports_align_corners = false;
            }
        } else if (dim == 4) {
            // 4D: nearest, bilinear, or bicubic
            uint8_t m = mode_selector % 3;
            if (m == 0) {
                mode_type = 0; // nearest
                mode_supports_align_corners = false;
            } else if (m == 1) {
                mode_type = 2; // bilinear
                mode_supports_align_corners = true;
            } else {
                mode_type = 3; // bicubic
                mode_supports_align_corners = true;
            }
        } else { // dim == 5
            // 5D: nearest or trilinear
            if (mode_selector & 0x1) {
                mode_type = 4; // trilinear
                mode_supports_align_corners = true;
            } else {
                mode_type = 0; // nearest
                mode_supports_align_corners = false;
            }
        }
        
        // Only use align_corners if mode supports it
        bool effective_align_corners = mode_supports_align_corners && align_corners;
        
        // Generate a valid scale factor (positive, reasonable range)
        double scale = 1.0;
        if (offset < Size) {
            scale = 0.5 + (Data[offset++] % 64) * 0.1; // Range: 0.5 to 6.8
        }
        
        // Helper lambda to create scale factor vector
        auto make_scale_vector = [](int64_t dim, double base_scale, const uint8_t* Data, size_t Size, size_t& offset) {
            std::vector<double> scale_factors;
            for (int i = 2; i < dim; i++) {
                double sf = base_scale;
                if (offset < Size) {
                    sf = 0.5 + (Data[offset++] % 64) * 0.1;
                }
                scale_factors.push_back(sf);
            }
            return scale_factors;
        };
        
        // Helper lambda to set mode on options
        auto apply_mode = [](torch::nn::UpsampleOptions& options, int mode_type) {
            switch (mode_type) {
                case 0: options.mode(torch::kNearest); break;
                case 1: options.mode(torch::kLinear); break;
                case 2: options.mode(torch::kBilinear); break;
                case 3: options.mode(torch::kBicubic); break;
                case 4: options.mode(torch::kTrilinear); break;
            }
        };
        
        // Test 1: Upsample with scale factor
        if (use_scale_factor) {
            try {
                std::vector<double> scale_factors = make_scale_vector(dim, scale, Data, Size, offset);
                
                auto options = torch::nn::UpsampleOptions()
                    .scale_factor(scale_factors);
                
                apply_mode(options, mode_type);
                
                if (mode_supports_align_corners) {
                    options.align_corners(effective_align_corners);
                }
                
                torch::nn::Upsample upsample(options);
                torch::Tensor output = upsample->forward(input);
            } catch (const std::exception &) {
                // Silently catch expected failures (e.g., dimension mismatches)
            }
        }
        
        // Test 2: Upsample with explicit size
        {
            try {
                std::vector<int64_t> target_sizes;
                for (int i = 2; i < dim; i++) {
                    int64_t current_size = input.size(i);
                    int64_t new_size = current_size;
                    if (offset < Size) {
                        // Generate size between 1 and 64
                        new_size = (Data[offset++] % 64) + 1;
                    } else {
                        new_size = std::max(int64_t(1), current_size * 2);
                    }
                    target_sizes.push_back(new_size);
                }
                
                auto options = torch::nn::UpsampleOptions()
                    .size(target_sizes);
                
                apply_mode(options, mode_type);
                
                if (mode_supports_align_corners) {
                    options.align_corners(effective_align_corners);
                }
                
                torch::nn::Upsample upsample(options);
                torch::Tensor output = upsample->forward(input);
            } catch (const std::exception &) {
                // Silently catch expected failures
            }
        }
        
        // Test 3: Single scalar scale factor (as vector with one element per spatial dim)
        {
            try {
                std::vector<double> scale_vec(dim - 2, scale);
                
                auto options = torch::nn::UpsampleOptions()
                    .scale_factor(scale_vec);
                
                apply_mode(options, mode_type);
                
                if (mode_supports_align_corners) {
                    options.align_corners(effective_align_corners);
                }
                
                torch::nn::Upsample upsample(options);
                torch::Tensor output = upsample->forward(input);
            } catch (const std::exception &) {
                // Silently catch expected failures
            }
        }
        
        // Test 4: Test nearest mode specifically (always valid)
        {
            try {
                std::vector<double> scale_vec(dim - 2, 2.0);
                
                auto options = torch::nn::UpsampleOptions()
                    .mode(torch::kNearest)
                    .scale_factor(scale_vec);
                
                torch::nn::Upsample upsample(options);
                torch::Tensor output = upsample->forward(input);
            } catch (const std::exception &) {
                // Silently catch expected failures
            }
        }
        
        // Test 5: Test dimension-specific interpolation modes
        {
            try {
                std::vector<double> scale_vec(dim - 2, scale);
                
                if (dim == 4) {
                    // Test bilinear with align_corners variations
                    auto options = torch::nn::UpsampleOptions()
                        .mode(torch::kBilinear)
                        .scale_factor(scale_vec)
                        .align_corners(effective_align_corners);
                    
                    torch::nn::Upsample upsample(options);
                    torch::Tensor output = upsample->forward(input);
                } else if (dim == 5) {
                    // Test trilinear
                    auto options = torch::nn::UpsampleOptions()
                        .mode(torch::kTrilinear)
                        .scale_factor(scale_vec)
                        .align_corners(effective_align_corners);
                    
                    torch::nn::Upsample upsample(options);
                    torch::Tensor output = upsample->forward(input);
                } else if (dim == 3) {
                    // Test linear
                    auto options = torch::nn::UpsampleOptions()
                        .mode(torch::kLinear)
                        .scale_factor(scale_vec)
                        .align_corners(effective_align_corners);
                    
                    torch::nn::Upsample upsample(options);
                    torch::Tensor output = upsample->forward(input);
                }
            } catch (const std::exception &) {
                // Silently catch expected failures
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