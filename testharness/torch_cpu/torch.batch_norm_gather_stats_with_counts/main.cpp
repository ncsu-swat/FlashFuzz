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
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensors for batch_norm_gather_stats_with_counts
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have more data to parse
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor mean = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor invstd = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse running_mean and running_var
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor running_mean = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor running_var = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create count tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor counts = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse momentum
        double momentum = 0.1;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Parse eps
        double eps = 1e-5;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Try to make counts tensor valid for the operation
        // Count tensor should be a 1D tensor with the same number of elements as the batch dimension
        if (counts.dim() != 1 && counts.numel() > 0) {
            counts = counts.reshape({counts.numel()});
        }
        
        // Try to make the tensors compatible with each other
        if (mean.dim() > 0 && invstd.dim() > 0) {
            // Try to ensure mean and invstd have the same shape
            if (mean.sizes() != invstd.sizes()) {
                // Reshape invstd to match mean's shape if possible
                if (invstd.numel() == mean.numel() && mean.numel() > 0) {
                    invstd = invstd.reshape(mean.sizes());
                }
            }
            
            // Try to ensure running_mean and running_var match mean's shape
            if (running_mean.numel() > 0 && running_mean.sizes() != mean.sizes() && running_mean.numel() == mean.numel()) {
                running_mean = running_mean.reshape(mean.sizes());
            }
            
            if (running_var.numel() > 0 && running_var.sizes() != mean.sizes() && running_var.numel() == mean.numel()) {
                running_var = running_var.reshape(mean.sizes());
            }
        }
        
        // Apply batch_norm_gather_stats_with_counts
        auto result = torch::batch_norm_gather_stats_with_counts(
            input, mean, invstd, running_mean, running_var, momentum, eps, counts);
        
        // Use the result to prevent optimization
        auto mean_result = std::get<0>(result);
        auto var_result = std::get<1>(result);
        
        // Perform some operation on the result to ensure it's used
        if (mean_result.numel() > 0 && var_result.numel() > 0) {
            auto sum = mean_result.sum() + var_result.sum();
            if (sum.item<float>() == -1.0f) {
                // This branch is unlikely to be taken, but prevents the compiler
                // from optimizing away our computation
                return 1;
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
