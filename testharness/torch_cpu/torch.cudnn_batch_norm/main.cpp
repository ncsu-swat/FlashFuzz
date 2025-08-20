#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Check if we have enough data
        if (Size < 5) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create weight and bias tensors
        torch::Tensor weight, bias;
        
        if (offset < Size) {
            weight = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create default weight tensor with same shape as input's first dimension
            if (input.dim() > 1) {
                weight = torch::ones({input.size(1)}, input.options());
            } else {
                weight = torch::ones({1}, input.options());
            }
        }
        
        if (offset < Size) {
            bias = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create default bias tensor with same shape as input's first dimension
            if (input.dim() > 1) {
                bias = torch::zeros({input.size(1)}, input.options());
            } else {
                bias = torch::zeros({1}, input.options());
            }
        }
        
        // Create running_mean and running_var tensors
        torch::Tensor running_mean, running_var;
        
        if (offset < Size) {
            running_mean = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create default running_mean tensor
            if (input.dim() > 1) {
                running_mean = torch::zeros({input.size(1)}, input.options());
            } else {
                running_mean = torch::zeros({1}, input.options());
            }
        }
        
        if (offset < Size) {
            running_var = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create default running_var tensor
            if (input.dim() > 1) {
                running_var = torch::ones({input.size(1)}, input.options());
            } else {
                running_var = torch::ones({1}, input.options());
            }
        }
        
        // Get training mode and momentum from input data
        bool training = true;
        double momentum = 0.1;
        double eps = 1e-5;
        
        if (offset + 1 < Size) {
            training = Data[offset++] % 2 == 0;
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Move tensors to CUDA if available
        if (torch::cuda::is_available()) {
            input = input.cuda();
            weight = weight.cuda();
            bias = bias.cuda();
            running_mean = running_mean.cuda();
            running_var = running_var.cuda();
        }
        
        // Apply cudnn_batch_norm
        auto result = torch::cudnn_batch_norm(
            input,
            weight,
            bias,
            running_mean,
            running_var,
            training,
            momentum,
            eps
        );
        
        // Get the output tensor
        torch::Tensor output = std::get<0>(result);
        
        // Ensure the output is moved back to CPU if it was on CUDA
        if (output.is_cuda()) {
            output = output.cpu();
        }
        
        // Perform some operation on the output to ensure it's used
        auto sum = output.sum().item<float>();
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