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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a byte to determine which activation to use
        uint8_t activation_type = 0;
        if (offset < Size) {
            activation_type = Data[offset++];
        }
        
        // Create scale and zero_point for quantization
        double scale = 0.1;
        int64_t zero_point = 0;
        
        if (offset + sizeof(double) + sizeof(int64_t) <= Size) {
            memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Ensure scale is positive and reasonable
        scale = std::abs(scale);
        if (scale < 1e-6) scale = 1e-6;
        if (scale > 1e6) scale = 1e6;
        
        // Ensure zero_point is in valid range for int8
        zero_point = std::max(std::min(zero_point, static_cast<int64_t>(127)), static_cast<int64_t>(-128));
        
        // Quantize the input tensor to qint8
        torch::Tensor q_input;
        try {
            q_input = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQInt8);
        } catch (...) {
            // If quantization fails, create a simple quantized tensor
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            auto simple_tensor = torch::ones({1, 1}, options);
            q_input = torch::quantize_per_tensor(simple_tensor, 0.1, 0, torch::kQInt8);
        }
        
        // Apply different activation functions based on activation_type
        switch (activation_type % 4) {
            case 0: {
                // ReLU - use functional interface for quantized tensors
                auto output = torch::relu(q_input);
                break;
            }
            case 1: {
                // ReLU6 - use functional interface for quantized tensors
                auto output = torch::relu6(q_input);
                break;
            }
            case 2: {
                // Hardtanh
                double min_val = -1.0;
                double max_val = 1.0;
                
                // Get min_val and max_val from input if available
                if (offset + 2 * sizeof(double) <= Size) {
                    memcpy(&min_val, Data + offset, sizeof(double));
                    offset += sizeof(double);
                    memcpy(&max_val, Data + offset, sizeof(double));
                    offset += sizeof(double);
                }
                
                // Ensure max_val >= min_val
                if (max_val < min_val) {
                    std::swap(min_val, max_val);
                }
                
                auto output = torch::hardtanh(q_input, min_val, max_val);
                break;
            }
            case 3: {
                // ELU
                double alpha = 1.0;
                
                // Get alpha from input if available
                if (offset + sizeof(double) <= Size) {
                    memcpy(&alpha, Data + offset, sizeof(double));
                    offset += sizeof(double);
                }
                
                // Ensure alpha is positive
                alpha = std::abs(alpha);
                if (alpha < 1e-6) alpha = 1e-6;
                
                auto output = torch::elu(q_input, alpha);
                break;
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
