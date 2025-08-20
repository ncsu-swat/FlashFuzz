#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for InstanceNorm1d from the remaining data
        uint8_t num_features = 0;
        float eps = 1e-5;
        float momentum = 0.1;
        bool affine = false;
        bool track_running_stats = false;
        
        if (offset < Size) {
            // Extract num_features - ensure it's at least 1
            num_features = Data[offset++] + 1;
        }
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure eps is positive and not too small
            eps = std::abs(eps);
            if (eps < 1e-10) eps = 1e-5;
        }
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure momentum is in [0, 1]
            momentum = std::abs(momentum);
            if (momentum > 1.0) momentum = momentum - std::floor(momentum);
        }
        
        if (offset < Size) {
            affine = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            track_running_stats = Data[offset++] & 0x1;
        }
        
        // Reshape input tensor if needed to match InstanceNorm1d requirements
        // InstanceNorm1d expects input of shape (N, C, L) or (C, L)
        auto input_sizes = input.sizes().vec();
        
        // If input is empty or scalar, reshape it to a valid shape
        if (input_sizes.empty()) {
            input = input.reshape({1, num_features, 1});
        } 
        // If input is 1D, reshape to (1, C, L)
        else if (input_sizes.size() == 1) {
            int64_t length = input_sizes[0];
            input = input.reshape({1, num_features, length > 0 ? length : 1});
        }
        // If input is 2D, interpret as (C, L)
        else if (input_sizes.size() == 2) {
            // Keep as is, InstanceNorm1d accepts (C, L)
        }
        // If input has more than 3 dimensions, reshape to 3D
        else if (input_sizes.size() > 3) {
            int64_t total_elements = input.numel();
            int64_t L = 1;
            int64_t N = 1;
            if (total_elements > 0) {
                L = total_elements / (num_features * N);
                if (L <= 0) L = 1;
            }
            input = input.reshape({N, num_features, L});
        }
        
        // Create InstanceNorm1d module
        torch::nn::InstanceNorm1d instance_norm(
            torch::nn::InstanceNorm1dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Apply InstanceNorm1d
        torch::Tensor output = instance_norm(input);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        
        // Convert to CPU if needed and access a value to ensure computation
        if (sum.device().is_cuda()) {
            sum = sum.to(torch::kCPU);
        }
        
        float result = sum.item<float>();
        
        // Use the result in a way that prevents the compiler from optimizing it away
        if (std::isnan(result) || std::isinf(result)) {
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