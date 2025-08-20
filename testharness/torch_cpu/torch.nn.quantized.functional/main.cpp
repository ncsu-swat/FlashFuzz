#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create scale and zero_point for quantization
        double scale = 0.1;
        int64_t zero_point = 0;
        
        if (offset + sizeof(double) + sizeof(int64_t) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure scale is positive and reasonable
            scale = std::abs(scale);
            if (scale < 1e-10) scale = 0.1;
            
            // Ensure zero_point is in valid range for int8
            zero_point = zero_point % 256;
        }
        
        // Get operation type from input data
        uint8_t op_type = 0;
        if (offset < Size) {
            op_type = Data[offset++] % 5; // Limit to 5 operations
        }
        
        // Quantize the input tensor
        torch::Tensor q_input;
        try {
            q_input = torch::quantize_per_tensor(input, scale, zero_point, torch::kQInt8);
        } catch (const std::exception&) {
            // If quantization fails, try with a different tensor
            input = torch::ones(input.sizes(), torch::kFloat);
            q_input = torch::quantize_per_tensor(input, scale, zero_point, torch::kQInt8);
        }
        
        // Apply different quantized functional operations based on op_type
        switch (op_type) {
            case 0: {
                // relu
                torch::Tensor result = torch::relu(q_input);
                break;
            }
            case 1: {
                // hardtanh
                double min_val = -1.0, max_val = 1.0;
                if (offset + 2 * sizeof(double) <= Size) {
                    std::memcpy(&min_val, Data + offset, sizeof(double));
                    offset += sizeof(double);
                    std::memcpy(&max_val, Data + offset, sizeof(double));
                    offset += sizeof(double);
                }
                torch::Tensor result = torch::hardtanh(q_input, min_val, max_val);
                break;
            }
            case 2: {
                // elu
                double alpha = 1.0;
                if (offset + sizeof(double) <= Size) {
                    std::memcpy(&alpha, Data + offset, sizeof(double));
                    offset += sizeof(double);
                    if (alpha <= 0) alpha = 1.0;
                }
                torch::Tensor result = torch::elu(q_input, alpha);
                break;
            }
            case 3: {
                // leaky_relu
                double negative_slope = 0.01;
                if (offset + sizeof(double) <= Size) {
                    std::memcpy(&negative_slope, Data + offset, sizeof(double));
                    offset += sizeof(double);
                    if (negative_slope <= 0) negative_slope = 0.01;
                }
                torch::nn::LeakyReLUOptions options(negative_slope);
                torch::Tensor result = torch::nn::functional::leaky_relu(q_input, options);
                break;
            }
            case 4: {
                // sigmoid
                torch::Tensor result = torch::sigmoid(q_input);
                break;
            }
            default:
                // Default to relu if op_type is out of range
                torch::Tensor result = torch::relu(q_input);
                break;
        }
        
        // Create a second tensor for binary operations
        if (offset < Size) {
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            try {
                torch::Tensor q_input2 = torch::quantize_per_tensor(input2, scale, zero_point, torch::kQInt8);
                
                // Try add operation
                torch::Tensor add_result = torch::add(q_input, q_input2);
                
                // Try mul operation
                torch::Tensor mul_result = torch::mul(q_input, q_input2);
                
                // Try cat operation if tensors have compatible dimensions
                if (q_input.dim() > 0 && q_input2.dim() > 0) {
                    int64_t dim = 0;
                    if (offset < Size) {
                        dim = static_cast<int64_t>(Data[offset++]) % std::max(q_input.dim(), q_input2.dim());
                    }
                    
                    std::vector<torch::Tensor> tensors = {q_input, q_input2};
                    torch::Tensor cat_result = torch::cat(tensors, dim);
                }
            } catch (const std::exception&) {
                // Ignore errors in binary operations
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