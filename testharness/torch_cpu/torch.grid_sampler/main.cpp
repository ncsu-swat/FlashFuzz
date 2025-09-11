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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create grid tensor
        torch::Tensor grid;
        if (offset < Size) {
            grid = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a compatible grid
            if (input.dim() >= 4) {
                auto batch_size = input.size(0);
                auto height = input.size(2);
                auto width = input.size(3);
                
                // Create a grid of appropriate shape for the input
                grid = torch::zeros({batch_size, height, width, 2}, input.options());
            } else {
                // For inputs with fewer dimensions, create a minimal grid
                grid = torch::zeros({1, 1, 1, 2}, input.options());
            }
        }
        
        // Parse interpolation mode from data
        int64_t interpolation_mode = 0; // bilinear
        if (offset < Size) {
            interpolation_mode = static_cast<int64_t>(Data[offset++]) % 3; // 0=bilinear, 1=nearest, 2=bicubic
        }
        
        // Parse padding mode from data
        int64_t padding_mode = 0; // zeros
        if (offset < Size) {
            padding_mode = static_cast<int64_t>(Data[offset++]) % 4; // 0=zeros, 1=border, 2=reflection, 3=circular
        }
        
        // Parse align_corners flag from data
        bool align_corners = false;
        if (offset < Size) {
            align_corners = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Apply grid_sampler operation
        torch::Tensor output = torch::grid_sampler(
            input, 
            grid, 
            interpolation_mode,
            padding_mode,
            align_corners
        );
        
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
