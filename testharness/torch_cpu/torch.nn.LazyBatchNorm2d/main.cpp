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
        
        // Need at least a few bytes to create meaningful input
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for BatchNorm2d from the remaining data
        int64_t num_features = 0;
        double eps = 1e-5;
        double momentum = 0.1;
        bool affine = true;
        bool track_running_stats = true;
        
        if (offset < Size) {
            num_features = static_cast<int64_t>(Data[offset++]);
            // Ensure num_features is at least 1
            num_features = std::max(int64_t(1), num_features);
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (offset < Size) {
            affine = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            track_running_stats = Data[offset++] & 0x1;
        }
        
        // Create BatchNorm2d module (LazyBatchNorm2d doesn't exist, use regular BatchNorm2d)
        torch::nn::BatchNorm2d batch_norm(
            torch::nn::BatchNorm2dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Apply the batch norm operation
        torch::Tensor output = batch_norm->forward(input);
        
        // Force computation to ensure any errors are triggered
        output = output.contiguous();
        
        // Access some elements to ensure computation happens
        if (output.numel() > 0) {
            float sum = output.sum().item<float>();
            (void)sum; // Prevent unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
