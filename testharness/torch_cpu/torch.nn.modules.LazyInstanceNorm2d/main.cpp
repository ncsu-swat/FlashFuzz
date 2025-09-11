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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 4 dimensions for InstanceNorm2d
        // If not, reshape it to have 4 dimensions
        if (input.dim() < 4) {
            std::vector<int64_t> new_shape;
            for (int i = 0; i < input.dim(); i++) {
                new_shape.push_back(input.size(i));
            }
            while (new_shape.size() < 4) {
                new_shape.push_back(1);
            }
            input = input.reshape(new_shape);
        }
        
        // Extract parameters for InstanceNorm2d
        uint8_t param_byte = 0;
        if (offset < Size) {
            param_byte = Data[offset++];
        }
        
        // Parse num_features from the input tensor
        int64_t num_features = input.size(1); // Channels dimension
        
        // Parse eps parameter (small value to avoid division by zero)
        double eps = 1e-5;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure eps is positive
            eps = std::abs(eps);
            if (eps == 0.0) eps = 1e-5;
        }
        
        // Parse momentum parameter
        double momentum = 0.1;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure momentum is between 0 and 1
            momentum = std::abs(momentum);
            if (momentum > 1.0) momentum = momentum - std::floor(momentum);
        }
        
        // Parse affine parameter
        bool affine = (param_byte & 0x01) != 0;
        
        // Parse track_running_stats parameter
        bool track_running_stats = (param_byte & 0x02) != 0;
        
        // Create InstanceNorm2d module
        torch::nn::InstanceNorm2d instance_norm(
            torch::nn::InstanceNorm2dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Apply the module to the input tensor
        torch::Tensor output = instance_norm(input);
        
        // Force computation to ensure any potential errors are triggered
        output = output.contiguous();
        
        // Access some elements to ensure computation is performed
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
