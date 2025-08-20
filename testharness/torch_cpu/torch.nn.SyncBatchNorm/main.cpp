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
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for BatchNorm (using regular BatchNorm as SyncBatchNorm is not available)
        // We need at least 4 bytes for the parameters
        if (offset + 4 >= Size) {
            return 0;
        }
        
        // Extract num_features from the input tensor
        int64_t num_features = 0;
        if (input.dim() >= 2) {
            num_features = input.size(1);
        } else if (input.dim() == 1) {
            num_features = input.size(0);
        } else {
            num_features = 1;
        }
        
        // Parse parameters for BatchNorm
        double eps = 1e-5;
        double momentum = 0.1;
        bool affine = true;
        bool track_running_stats = true;
        
        if (offset < Size) {
            // Use a byte to determine eps (small positive value)
            uint8_t eps_byte = Data[offset++];
            eps = static_cast<double>(eps_byte) / 255.0 * 0.1 + 1e-10;
        }
        
        if (offset < Size) {
            // Use a byte to determine momentum (between 0 and 1)
            uint8_t momentum_byte = Data[offset++];
            momentum = static_cast<double>(momentum_byte) / 255.0;
        }
        
        if (offset < Size) {
            // Use a byte to determine boolean parameters
            uint8_t bool_byte = Data[offset++];
            affine = (bool_byte & 0x1);
            track_running_stats = (bool_byte & 0x2);
        }
        
        // Create BatchNorm module (using regular BatchNorm as fallback)
        torch::nn::BatchNorm1d sync_bn(
            torch::nn::BatchNorm1dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Apply BatchNorm to the input tensor
        torch::Tensor output;
        
        // BatchNorm expects input of shape [N, C, ...] where N is batch size and C is num_features
        // If input doesn't have at least 2 dimensions, we need to reshape it
        if (input.dim() < 2) {
            if (input.dim() == 0) {
                // Scalar tensor, reshape to [1, 1]
                input = input.reshape({1, 1});
            } else if (input.dim() == 1) {
                // 1D tensor, reshape to [1, C]
                input = input.reshape({1, input.size(0)});
            }
        }
        
        // Ensure input has float-like dtype for BatchNorm
        if (input.dtype() == torch::kInt8 || 
            input.dtype() == torch::kUInt8 || 
            input.dtype() == torch::kInt16 || 
            input.dtype() == torch::kInt32 || 
            input.dtype() == torch::kInt64 ||
            input.dtype() == torch::kBool) {
            input = input.to(torch::kFloat);
        }
        
        // Forward pass
        output = sync_bn(input);
        
        // Try to access some properties to ensure the module works
        auto running_mean = sync_bn->running_mean;
        auto running_var = sync_bn->running_var;
        
        if (affine) {
            auto weight = sync_bn->weight;
            auto bias = sync_bn->bias;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}