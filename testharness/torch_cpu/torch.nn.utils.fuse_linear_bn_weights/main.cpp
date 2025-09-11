#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create linear weight tensor
        torch::Tensor linear_weight = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create linear bias tensor (can be empty)
        torch::Tensor linear_bias;
        if (offset < Size) {
            linear_bias = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            linear_bias = torch::Tensor();
        }
        
        // Create bn_running_mean tensor
        torch::Tensor bn_running_mean;
        if (offset < Size) {
            bn_running_mean = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            bn_running_mean = torch::Tensor();
        }
        
        // Create bn_running_var tensor
        torch::Tensor bn_running_var;
        if (offset < Size) {
            bn_running_var = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            bn_running_var = torch::Tensor();
        }
        
        // Create bn_weight tensor
        torch::Tensor bn_weight;
        if (offset < Size) {
            bn_weight = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            bn_weight = torch::Tensor();
        }
        
        // Create bn_bias tensor
        torch::Tensor bn_bias;
        if (offset < Size) {
            bn_bias = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            bn_bias = torch::Tensor();
        }
        
        // Create eps value
        double eps = 1e-5;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure eps is positive
            if (eps <= 0) {
                eps = 1e-5;
            }
        }
        
        // Call fuse_linear_bn_weights
        auto result = torch::fuse_linear_bn_weights(
            linear_weight,
            linear_bias,
            bn_running_mean,
            bn_running_var,
            bn_weight,
            bn_bias,
            eps
        );
        
        // Unpack the result
        torch::Tensor fused_weight = std::get<0>(result);
        torch::Tensor fused_bias = std::get<1>(result);
        
        // Perform some operation on the result to ensure it's used
        if (fused_weight.defined() && fused_bias.defined()) {
            auto sum_weight = torch::sum(fused_weight);
            auto sum_bias = torch::sum(fused_bias);
            auto total_sum = sum_weight + sum_bias;
            
            // Just to make sure the tensors are actually used
            if (total_sum.item<float>() == 0.0f) {
                // This is just to use the result, not actually important
                torch::Tensor dummy = torch::ones({1});
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
