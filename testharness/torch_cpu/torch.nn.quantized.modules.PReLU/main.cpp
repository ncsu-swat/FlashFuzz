#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Early exit if not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            return 0;
        }
        
        // Ensure tensor is quantized
        if (!input_tensor.is_quantized()) {
            // Quantize the tensor to qint8
            auto scale = 1.0f / 128.0f;
            auto zero_point = 0;
            
            // Try to quantize the tensor
            try {
                input_tensor = torch::quantize_per_tensor(
                    input_tensor.to(torch::kFloat), 
                    scale, 
                    zero_point, 
                    torch::kQInt8
                );
            } catch (const std::exception& e) {
                return 0;
            }
        }
        
        // Create weight parameter for PReLU
        torch::Tensor weight;
        try {
            // Create a weight tensor with the same number of channels as input
            int64_t num_params = 1;
            if (input_tensor.dim() > 1) {
                num_params = input_tensor.size(1);
            }
            
            // Create a small weight tensor
            if (offset + 1 < Size) {
                // Use some bytes from the input data to determine if we want a single parameter
                bool single_param = (Data[offset++] % 2 == 0);
                if (single_param) {
                    num_params = 1;
                }
            }
            
            // Create weight tensor
            std::vector<float> weight_data(num_params);
            for (int64_t i = 0; i < num_params; i++) {
                if (offset < Size) {
                    // Use input data to generate weight values
                    weight_data[i] = static_cast<float>(Data[offset++]) / 255.0f;
                } else {
                    weight_data[i] = 0.25f; // Default value
                }
            }
            
            weight = torch::tensor(weight_data, torch::kFloat);
        } catch (const std::exception& e) {
            return 0;
        }
        
        // Create quantized PReLU module
        try {
            auto scale = 1.0f / 128.0f;
            auto zero_point = 0;
            
            // Quantize the weight tensor
            auto quantized_weight = torch::quantize_per_tensor(
                weight, 
                scale, 
                zero_point, 
                torch::kQInt8
            );
            
            // Create the PReLU module using functional approach
            auto output = torch::nn::functional::prelu(input_tensor, quantized_weight);
            
            // Verify output is not empty
            if (output.numel() != input_tensor.numel()) {
                throw std::runtime_error("Output tensor has different number of elements than input");
            }
            
            // Try to access some elements to ensure the tensor is valid
            if (output.numel() > 0) {
                auto item = output.item();
            }
        } catch (const std::exception& e) {
            // Catch exceptions from PReLU operation
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
