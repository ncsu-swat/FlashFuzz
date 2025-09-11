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
        if (input.dim() < 4) {
            // Expand dimensions to make it 4D
            while (input.dim() < 4) {
                input = input.unsqueeze(0);
            }
        }
        
        // Extract parameters for InstanceNorm2d
        int64_t num_features = input.size(1); // Number of channels
        
        // Parse additional parameters from the input data
        bool affine = offset < Size ? (Data[offset++] & 0x1) : false;
        bool track_running_stats = offset < Size ? (Data[offset++] & 0x1) : false;
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
        
        // Create InstanceNorm2d module (LazyInstanceNorm2d doesn't exist, use regular InstanceNorm2d)
        torch::nn::InstanceNorm2d instance_norm(
            torch::nn::InstanceNorm2dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Apply the module to the input tensor
        torch::Tensor output = instance_norm->forward(input);
        
        // Force computation to ensure lazy tensors are materialized
        output = output.clone();
        
        // Access some properties to ensure computation is done
        auto output_size = output.sizes();
        auto output_dtype = output.dtype();
        
        // Check if output has NaN or Inf values
        bool has_nan = torch::isnan(output).any().item<bool>();
        bool has_inf = torch::isinf(output).any().item<bool>();
        
        if (has_nan || has_inf) {
            // This is not an error, just interesting information
            // We don't throw an exception to allow the fuzzer to continue
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
