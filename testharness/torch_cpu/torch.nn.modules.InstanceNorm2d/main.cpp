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
        
        // Skip if we don't have enough data
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 4 dimensions for InstanceNorm2d (N, C, H, W)
        if (input.dim() < 3) {
            // Reshape to 4D if needed
            std::vector<int64_t> new_shape;
            if (input.dim() == 0) {
                // Scalar to 4D
                new_shape = {1, 1, 1, 1};
            } else if (input.dim() == 1) {
                // 1D to 4D
                new_shape = {1, input.size(0), 1, 1};
            } else if (input.dim() == 2) {
                // 2D to 4D
                new_shape = {1, input.size(0), input.size(1), 1};
            }
            input = input.reshape(new_shape);
        }
        
        // Get number of channels (second dimension)
        int64_t num_channels = input.size(1);
        if (num_channels == 0) {
            num_channels = 1;
            // Reshape to have at least one channel
            std::vector<int64_t> new_shape = input.sizes().vec();
            new_shape[1] = 1;
            input = input.reshape(new_shape);
        }
        
        // Parse configuration parameters from the input data
        bool affine = offset < Size ? (Data[offset++] & 0x1) : true;
        bool track_running_stats = offset < Size ? (Data[offset++] & 0x1) : true;
        double eps = 1e-5;
        double momentum = 0.1;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure eps is positive
            eps = std::abs(eps);
            if (eps == 0.0) eps = 1e-5;
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure momentum is between 0 and 1
            momentum = std::abs(momentum);
            if (momentum > 1.0) momentum = momentum - std::floor(momentum);
        }
        
        // Create InstanceNorm2d module
        torch::nn::InstanceNorm2d instance_norm(
            torch::nn::InstanceNorm2dOptions(num_channels)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Convert input to float if it's not already a floating point type
        if (!torch::isFloatingType(input.scalar_type())) {
            input = input.to(torch::kFloat);
        }
        
        // Apply InstanceNorm2d
        torch::Tensor output = instance_norm->forward(input);
        
        // Test with eval mode
        instance_norm->eval();
        torch::Tensor output_eval = instance_norm->forward(input);
        
        // Test with train mode again
        instance_norm->train();
        torch::Tensor output_train = instance_norm->forward(input);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
