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
        
        // Need at least some data to proceed
        if (Size < 5) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create weight, bias, mean, and var tensors
        // These should have the same size as the number of channels (dim 1) in the input
        int64_t num_channels = 1;
        if (input.dim() > 1) {
            num_channels = input.size(1);
        }
        
        // Create weight tensor
        torch::Tensor weight;
        if (offset < Size) {
            weight = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure weight has correct shape (should be 1D with size = num_channels)
            if (weight.dim() != 1 || weight.size(0) != num_channels) {
                weight = torch::ones({num_channels});
            }
        } else {
            weight = torch::ones({num_channels});
        }
        
        // Create bias tensor
        torch::Tensor bias;
        if (offset < Size) {
            bias = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure bias has correct shape
            if (bias.dim() != 1 || bias.size(0) != num_channels) {
                bias = torch::zeros({num_channels});
            }
        } else {
            bias = torch::zeros({num_channels});
        }
        
        // Create mean tensor
        torch::Tensor mean;
        if (offset < Size) {
            mean = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure mean has correct shape
            if (mean.dim() != 1 || mean.size(0) != num_channels) {
                mean = torch::zeros({num_channels});
            }
        } else {
            mean = torch::zeros({num_channels});
        }
        
        // Create var tensor
        torch::Tensor var;
        if (offset < Size) {
            var = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure var has correct shape
            if (var.dim() != 1 || var.size(0) != num_channels) {
                var = torch::ones({num_channels});
            }
            // Ensure var is positive (required for batch_norm)
            var = torch::abs(var) + 1e-5;
        } else {
            var = torch::ones({num_channels});
        }
        
        // Get epsilon value from the input data
        float eps = 1e-5;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure epsilon is positive and not too large
            eps = std::abs(eps);
            if (eps > 1.0f) eps = 1.0f;
            if (eps < 1e-10f) eps = 1e-10f;
        }
        
        // Apply batch_norm_elemt
        torch::Tensor output = torch::batch_norm_elemt(input, weight, bias, mean, var, eps);
        
        // Try to access the output to ensure computation is not optimized away
        float sum = output.sum().item<float>();
        if (std::isnan(sum) || std::isinf(sum)) {
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
