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
        
        // Need at least some data to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the input tensor has at least 2 dimensions for batch norm
        // First dimension is batch size, second is features
        if (input.dim() < 2) {
            // Reshape to at least 2D if needed
            std::vector<int64_t> new_shape;
            if (input.dim() == 0) {
                // Scalar tensor, reshape to [1, 1]
                new_shape = {1, 1};
            } else if (input.dim() == 1) {
                // 1D tensor, reshape to [1, size]
                new_shape = {1, input.size(0)};
            }
            
            if (!new_shape.empty()) {
                input = input.reshape(new_shape);
            }
        }
        
        // Get a value for epsilon from the input data
        double epsilon = 1e-5; // Default value
        if (offset + sizeof(float) <= Size) {
            float eps_val;
            std::memcpy(&eps_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure epsilon is positive and not too small
            if (std::isfinite(eps_val) && eps_val > 0) {
                epsilon = eps_val;
            }
        }
        
        // Call batch_norm_stats
        auto result = torch::batch_norm_stats(input, epsilon);
        
        // Unpack the result (mean and variance)
        auto mean = std::get<0>(result);
        auto var = std::get<1>(result);
        
        // Perform some operation with the results to ensure they're used
        auto sum = mean.sum() + var.sum();
        if (std::isnan(sum.item<double>())) {
            // This is a valid result in some cases, just return
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
