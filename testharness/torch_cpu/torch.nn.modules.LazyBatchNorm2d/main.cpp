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
        
        // Ensure input has at least 4 dimensions for BatchNorm2d (N, C, H, W)
        if (input.dim() < 4) {
            // Expand dimensions to make it 4D
            while (input.dim() < 4) {
                input = input.unsqueeze(0);
            }
        }
        
        // Extract parameters for LazyBatchNorm2d
        uint8_t num_features_byte = 0;
        if (offset < Size) {
            num_features_byte = Data[offset++];
        }
        
        // Get number of features from the second dimension (channels)
        int64_t num_features = input.size(1);
        
        // Extract other parameters
        double eps = 1e-5;
        double momentum = 0.1;
        bool affine = true;
        bool track_running_stats = true;
        
        if (offset + 3 < Size) {
            // Use some bytes to determine eps (small positive value)
            uint32_t eps_raw = 0;
            std::memcpy(&eps_raw, Data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            eps = std::max(1e-10, static_cast<double>(eps_raw) / std::numeric_limits<uint32_t>::max());
            
            // Use a byte to determine momentum (between 0 and 1)
            if (offset < Size) {
                momentum = static_cast<double>(Data[offset++]) / 255.0;
            }
            
            // Use bytes to determine boolean parameters
            if (offset < Size) {
                affine = Data[offset++] % 2 == 0;
            }
            if (offset < Size) {
                track_running_stats = Data[offset++] % 2 == 0;
            }
        }
        
        // Create BatchNorm2d module (LazyBatchNorm2d is not available, use regular BatchNorm2d)
        torch::nn::BatchNorm2d bn(
            torch::nn::BatchNorm2dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Apply the module to the input tensor
        torch::Tensor output = bn->forward(input);
        
        // Test the module in training and evaluation modes
        bn->train();
        torch::Tensor output_train = bn->forward(input);
        
        bn->eval();
        torch::Tensor output_eval = bn->forward(input);
        
        // Test with different input shapes
        if (offset + 2 < Size) {
            // Create a new input with different spatial dimensions
            uint8_t height_byte = Data[offset++];
            uint8_t width_byte = Data[offset++];
            
            int64_t new_height = 1 + (height_byte % 32);
            int64_t new_width = 1 + (width_byte % 32);
            
            // Create a new input tensor with the same batch size and channels
            // but different spatial dimensions
            torch::Tensor new_input;
            try {
                new_input = torch::ones({input.size(0), input.size(1), new_height, new_width});
                torch::Tensor new_output = bn->forward(new_input);
            } catch (const std::exception& e) {
                // Ignore exceptions from this test
            }
        }
        
        // Test with zero batch size
        try {
            torch::Tensor zero_batch_input = torch::ones({0, input.size(1), input.size(2), input.size(3)});
            torch::Tensor zero_batch_output = bn->forward(zero_batch_input);
        } catch (const std::exception& e) {
            // Ignore exceptions from this test
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}