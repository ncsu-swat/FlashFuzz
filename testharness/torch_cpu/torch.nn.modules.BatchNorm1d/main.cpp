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
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 2 dimensions for BatchNorm1d
        // If not, reshape it to have a batch dimension and a feature dimension
        if (input.dim() < 2) {
            if (input.dim() == 0) {
                // Scalar tensor, reshape to [1, 1]
                input = input.reshape({1, 1});
            } else if (input.dim() == 1) {
                // 1D tensor, reshape to [1, size]
                input = input.reshape({1, input.size(0)});
            }
        }
        
        // Extract parameters for BatchNorm1d from the remaining data
        int64_t num_features = input.size(1); // Use the feature dimension size
        
        // Parse additional parameters if data available
        bool affine = true;
        bool track_running_stats = true;
        double momentum = 0.1;
        double eps = 1e-5;
        
        if (offset + 4 <= Size) {
            // Use remaining bytes to determine parameters
            affine = (Data[offset++] % 2) == 0;
            track_running_stats = (Data[offset++] % 2) == 0;
            
            // Parse momentum (0.0 to 1.0)
            if (offset < Size) {
                momentum = static_cast<double>(Data[offset++]) / 255.0;
            }
            
            // Parse eps (small positive value)
            if (offset < Size) {
                eps = std::max(1e-10, static_cast<double>(Data[offset++]) / 1e4);
            }
        }
        
        // Create BatchNorm1d module
        torch::nn::BatchNorm1d bn(torch::nn::BatchNorm1dOptions(num_features)
                                  .affine(affine)
                                  .track_running_stats(track_running_stats)
                                  .momentum(momentum)
                                  .eps(eps));
        
        // Set module to evaluation mode with 50% probability if we have more data
        if (offset < Size && (Data[offset++] % 2) == 0) {
            bn->eval();
        } else {
            bn->train();
        }
        
        // Apply BatchNorm1d to the input tensor
        torch::Tensor output = bn->forward(input);
        
        // Try to access output properties to ensure computation completed
        auto output_size = output.sizes();
        auto output_dtype = output.dtype();
        
        // If running in training mode, try a backward pass if we have enough data
        if (bn->is_training() && offset < Size && (Data[offset++] % 2) == 0) {
            output.sum().backward();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
