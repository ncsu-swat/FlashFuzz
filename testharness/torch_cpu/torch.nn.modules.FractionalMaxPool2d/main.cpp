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
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (N, C, H, W) for FractionalMaxPool2d
        if (input.dim() < 3) {
            // Expand dimensions if needed
            while (input.dim() < 3) {
                input = input.unsqueeze(0);
            }
            // Add one more dimension if needed to make it 4D (N, C, H, W)
            if (input.dim() == 3) {
                input = input.unsqueeze(0);
            }
        }
        
        // Ensure H and W dimensions are large enough (at least 2)
        auto sizes = input.sizes();
        auto H = sizes[sizes.size() - 2];
        auto W = sizes[sizes.size() - 1];
        
        if (H < 2 || W < 2) {
            // Resize to ensure minimum dimensions
            std::vector<int64_t> new_sizes(sizes.begin(), sizes.end());
            if (H < 2) new_sizes[sizes.size() - 2] = 2;
            if (W < 2) new_sizes[sizes.size() - 1] = 2;
            input = input.resize_(new_sizes);
        }
        
        // Parse parameters for FractionalMaxPool2d
        if (offset + 8 > Size) {
            return 0;
        }
        
        // Parse output_size or output_ratio
        bool use_output_size = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
        
        // Parse kernel_size
        int64_t kernel_h = 0, kernel_w = 0;
        if (use_output_size) {
            // Parse output_size
            kernel_h = (offset < Size) ? (Data[offset++] % (H - 1)) + 1 : 1;
            kernel_w = (offset < Size) ? (Data[offset++] % (W - 1)) + 1 : 1;
        } else {
            // Parse output_ratio
            float ratio_h = 0.5, ratio_w = 0.5;
            if (offset + sizeof(float) <= Size) {
                std::memcpy(&ratio_h, Data + offset, sizeof(float));
                offset += sizeof(float);
                // Ensure ratio is between 0 and 1
                ratio_h = std::abs(ratio_h);
                ratio_h = ratio_h - std::floor(ratio_h);
                if (ratio_h < 0.1) ratio_h = 0.1;
                if (ratio_h > 0.9) ratio_h = 0.9;
            }
            
            if (offset + sizeof(float) <= Size) {
                std::memcpy(&ratio_w, Data + offset, sizeof(float));
                offset += sizeof(float);
                // Ensure ratio is between 0 and 1
                ratio_w = std::abs(ratio_w);
                ratio_w = ratio_w - std::floor(ratio_w);
                if (ratio_w < 0.1) ratio_w = 0.1;
                if (ratio_w > 0.9) ratio_w = 0.9;
            }
            
            kernel_h = static_cast<int64_t>(H * ratio_h);
            kernel_w = static_cast<int64_t>(W * ratio_w);
            
            // Ensure kernel size is at least 1
            if (kernel_h < 1) kernel_h = 1;
            if (kernel_w < 1) kernel_w = 1;
        }
        
        // Parse return_indices
        bool return_indices = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
        
        // Create FractionalMaxPool2d module
        torch::nn::FractionalMaxPool2dOptions options({kernel_h, kernel_w});
        
        auto pool = torch::nn::FractionalMaxPool2d(options);
        
        // Apply the pooling operation
        auto output = pool->forward(input);
        
        // Perform some operation with output to ensure it's used
        auto sum = output.sum();
        if (sum.item<float>() == -1.0f) {
            // This is just to use the result and avoid compiler optimizations
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
