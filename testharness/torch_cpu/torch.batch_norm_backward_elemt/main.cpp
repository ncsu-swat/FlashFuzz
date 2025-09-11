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
        
        // Need at least some data to create tensors
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensors for batch_norm_backward_elemt
        torch::Tensor grad_out = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create input, mean, invstd, weight, sum_dy, sum_dy_xmu, count
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor mean = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor invstd = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Optional weight tensor
        bool use_weight = offset < Size && (Data[offset++] & 0x1);
        torch::Tensor weight;
        if (use_weight) {
            weight = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Create sum_dy, sum_dy_xmu, count
        torch::Tensor sum_dy = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor sum_dy_xmu = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor count = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply batch_norm_backward_elemt operation
        torch::Tensor result;
        if (use_weight) {
            result = torch::batch_norm_backward_elemt(
                grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
        } else {
            result = torch::batch_norm_backward_elemt(
                grad_out, input, mean, invstd, {}, sum_dy, sum_dy_xmu, count);
        }
        
        // Perform some operation on the result to ensure it's used
        auto sum = result.sum();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
