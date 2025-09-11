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
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor - must be 5D for InstanceNorm3d (N, C, D, H, W)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor has 5 dimensions for InstanceNorm3d
        // If not, reshape it to have 5 dimensions
        if (input.dim() != 5) {
            std::vector<int64_t> new_shape;
            
            if (input.dim() > 5) {
                // Flatten extra dimensions into the last dimension
                new_shape = {1, 1, 1, 1, input.numel()};
            } else {
                // Add missing dimensions
                new_shape = {1, 1, 1, 1, 1};
                for (int i = 0; i < input.dim(); i++) {
                    new_shape[5 - input.dim() + i] = input.size(i);
                }
            }
            
            input = input.reshape(new_shape);
        }
        
        // Extract parameters for InstanceNorm3d from the input data
        bool affine = offset < Size ? (Data[offset++] % 2 == 0) : false;
        bool track_running_stats = offset < Size ? (Data[offset++] % 2 == 0) : false;
        double eps = 1e-5;
        double momentum = 0.1;
        
        if (offset + sizeof(float) <= Size) {
            float eps_raw;
            std::memcpy(&eps_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure eps is positive and not too small
            eps = std::abs(eps_raw);
            if (eps < 1e-10) eps = 1e-5;
        }
        
        if (offset + sizeof(float) <= Size) {
            float momentum_raw;
            std::memcpy(&momentum_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure momentum is in [0, 1]
            momentum = std::abs(momentum_raw);
            momentum = momentum > 1.0 ? momentum - std::floor(momentum) : momentum;
        }
        
        // Get the number of channels (2nd dimension)
        int64_t num_features = input.size(1);
        if (num_features == 0) {
            num_features = 1; // Ensure at least one channel
            input = input.reshape({input.size(0), 1, input.size(2), input.size(3), input.size(4)});
        }
        
        // Create InstanceNorm3d module
        torch::nn::InstanceNorm3d instance_norm(
            torch::nn::InstanceNorm3dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats));
        
        // Apply InstanceNorm3d
        torch::Tensor output = instance_norm->forward(input);
        
        // Try to access output properties to ensure computation completed
        auto output_size = output.sizes();
        auto output_dtype = output.dtype();
        
        // Try different training modes
        instance_norm->eval();
        torch::Tensor output_eval = instance_norm->forward(input);
        
        instance_norm->train();
        torch::Tensor output_train = instance_norm->forward(input);
        
        // Test with different input types if possible
        if (offset < Size) {
            auto dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Try to convert input to another dtype
            try {
                auto input_converted = input.to(dtype);
                auto output_converted = instance_norm->forward(input_converted);
            } catch (const std::exception&) {
                // Some dtype conversions might not be valid, ignore errors
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
