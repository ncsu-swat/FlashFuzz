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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions (N, C, L) for ConvBnReLU1d
        if (input.dim() < 3) {
            input = input.reshape({1, 1, input.numel()});
        }
        
        // Extract parameters for ConvBnReLU1d
        int64_t in_channels = input.size(1);
        int64_t out_channels = 1;
        int64_t kernel_size = 3;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        bool bias = true;
        
        // If we have more data, use it to set parameters
        if (offset + 7 <= Size) {
            out_channels = (Data[offset] % 8) + 1;
            kernel_size = (Data[offset + 1] % 5) + 1;
            stride = (Data[offset + 2] % 3) + 1;
            padding = Data[offset + 3] % 3;
            dilation = (Data[offset + 4] % 2) + 1;
            bias = Data[offset + 5] % 2 == 0;
            offset += 6;
        }
        
        // Create Conv1d module
        auto conv = torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                                     .stride(stride)
                                     .padding(padding)
                                     .dilation(dilation)
                                     .bias(bias));
        
        // Create BatchNorm1d module
        auto bn = torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(out_channels));
        
        // Create ReLU module
        auto relu = torch::nn::ReLU();
        
        // Initialize conv weights with values from the fuzzer data
        if (offset < Size) {
            auto weight_tensor = conv->weight;
            auto weight_size = weight_tensor.numel();
            
            std::vector<float> weight_data(weight_size);
            for (int i = 0; i < weight_size && offset < Size; i++) {
                weight_data[i] = static_cast<float>(Data[offset++]) / 255.0f;
            }
            
            weight_tensor.copy_(torch::from_blob(weight_data.data(), weight_tensor.sizes(), 
                                                torch::TensorOptions().dtype(torch::kFloat)));
        }
        
        // Initialize conv bias if used
        if (bias && conv->bias.defined() && offset < Size) {
            auto bias_tensor = conv->bias;
            auto bias_size = bias_tensor.numel();
            
            std::vector<float> bias_data(bias_size);
            for (int i = 0; i < bias_size && offset < Size; i++) {
                bias_data[i] = static_cast<float>(Data[offset++]) / 255.0f;
            }
            
            bias_tensor.copy_(torch::from_blob(bias_data.data(), bias_tensor.sizes(), 
                                              torch::TensorOptions().dtype(torch::kFloat)));
        }
        
        // Initialize batch norm parameters
        if (offset + 4 < Size) {
            // Running mean
            auto running_mean = bn->running_mean;
            auto mean_size = running_mean.numel();
            std::vector<float> mean_data(mean_size);
            for (int i = 0; i < mean_size && offset < Size; i++) {
                mean_data[i] = static_cast<float>(Data[offset++]) / 255.0f;
            }
            running_mean.copy_(torch::from_blob(mean_data.data(), running_mean.sizes(), 
                                               torch::TensorOptions().dtype(torch::kFloat)));
            
            // Running var
            auto running_var = bn->running_var;
            auto var_size = running_var.numel();
            std::vector<float> var_data(var_size);
            for (int i = 0; i < var_size && offset < Size; i++) {
                // Ensure variance is positive
                var_data[i] = static_cast<float>(Data[offset++]) / 255.0f + 0.01f;
            }
            running_var.copy_(torch::from_blob(var_data.data(), running_var.sizes(), 
                                              torch::TensorOptions().dtype(torch::kFloat)));
            
            // Weight and bias
            if (bn->weight.defined() && offset < Size) {
                auto weight = bn->weight;
                auto weight_size = weight.numel();
                std::vector<float> weight_data(weight_size);
                for (int i = 0; i < weight_size && offset < Size; i++) {
                    weight_data[i] = static_cast<float>(Data[offset++]) / 255.0f;
                }
                weight.copy_(torch::from_blob(weight_data.data(), weight.sizes(), 
                                             torch::TensorOptions().dtype(torch::kFloat)));
            }
            
            if (bn->bias.defined() && offset < Size) {
                auto bias = bn->bias;
                auto bias_size = bias.numel();
                std::vector<float> bias_data(bias_size);
                for (int i = 0; i < bias_size && offset < Size; i++) {
                    bias_data[i] = static_cast<float>(Data[offset++]) / 255.0f;
                }
                bias.copy_(torch::from_blob(bias_data.data(), bias.sizes(), 
                                           torch::TensorOptions().dtype(torch::kFloat)));
            }
        }
        
        // Set training mode based on fuzzer data
        if (offset < Size) {
            bool training_mode = Data[offset++] % 2 == 0;
            conv->train(training_mode);
            bn->train(training_mode);
        }
        
        // Apply the ConvBnReLU1d operation (conv -> bn -> relu)
        torch::Tensor output = conv->forward(input);
        output = bn->forward(output);
        output = relu->forward(output);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
