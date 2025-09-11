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
        
        // Need at least a few bytes for basic operations
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for Upsample from the remaining data
        bool align_corners = false;
        bool scale_factors_mode = false;
        
        if (offset < Size) {
            align_corners = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            scale_factors_mode = Data[offset++] & 0x1;
        }
        
        // Get input tensor dimensions
        auto input_sizes = input_tensor.sizes();
        int64_t dim = input_sizes.size();
        
        // Upsample requires at least 3D tensor (batch, channels, spatial dims)
        if (dim < 3) {
            // For tensors with fewer dimensions, add dimensions to make it at least 3D
            std::vector<int64_t> new_sizes;
            for (int i = 0; i < input_sizes.size(); i++) {
                new_sizes.push_back(input_sizes[i]);
            }
            
            while (new_sizes.size() < 3) {
                new_sizes.push_back(1);
            }
            
            input_tensor = input_tensor.reshape(new_sizes);
            input_sizes = input_tensor.sizes();
            dim = input_sizes.size();
        }
        
        // Create output size or scale factors
        std::vector<double> scale_factors;
        std::vector<int64_t> output_size;
        
        if (scale_factors_mode) {
            // Use scale factors
            for (int i = 2; i < dim; i++) {
                double scale = 1.0;
                if (offset + sizeof(double) <= Size) {
                    memcpy(&scale, Data + offset, sizeof(double));
                    offset += sizeof(double);
                }
                scale_factors.push_back(scale);
            }
        } else {
            // Use output size
            for (int i = 2; i < dim; i++) {
                int64_t size = 1;
                if (offset + sizeof(int64_t) <= Size) {
                    memcpy(&size, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    // Ensure size is positive
                    size = std::abs(size) % 100 + 1;
                }
                output_size.push_back(size);
            }
        }
        
        // Create Upsample module
        torch::nn::UpsampleOptions options;
        
        if (scale_factors_mode && !scale_factors.empty()) {
            options = torch::nn::UpsampleOptions()
                .align_corners(align_corners)
                .scale_factor(scale_factors);
        } else if (!output_size.empty()) {
            options = torch::nn::UpsampleOptions()
                .align_corners(align_corners)
                .size(output_size);
        } else {
            // Default to some output size if neither was properly set
            std::vector<int64_t> default_size;
            for (int i = 2; i < dim; i++) {
                default_size.push_back(input_sizes[i] * 2);
            }
            options = torch::nn::UpsampleOptions()
                .align_corners(align_corners)
                .size(default_size);
        }
        
        // Set interpolation mode based on remaining data
        if (offset < Size) {
            uint8_t mode_selector = Data[offset++] % 4;
            switch (mode_selector) {
                case 0:
                    options = options.mode(torch::kNearest);
                    break;
                case 1:
                    options = options.mode(torch::kLinear);
                    break;
                case 2:
                    options = options.mode(torch::kBilinear);
                    break;
                case 3:
                    options = options.mode(torch::kTrilinear);
                    break;
                default:
                    options = options.mode(torch::kNearest);
            }
        }
        
        // Create and apply the Upsample module
        torch::nn::Upsample upsample(options);
        torch::Tensor output = upsample->forward(input_tensor);
        
        // Ensure the output is used to prevent optimization
        auto sum = output.sum().item<float>();
        if (std::isnan(sum) || std::isinf(sum)) {
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
