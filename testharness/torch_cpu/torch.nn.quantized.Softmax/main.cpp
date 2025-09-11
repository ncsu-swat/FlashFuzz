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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a byte for dim parameter
        int64_t dim = 0;
        if (offset < Size) {
            dim = static_cast<int64_t>(Data[offset++]) % std::max(1, static_cast<int>(input_tensor.dim()));
        }
        
        // Create scale and zero_point for quantization
        double scale = 1.0;
        int64_t zero_point = 0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            scale = std::abs(scale);
            if (scale == 0.0) scale = 1.0;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            zero_point = zero_point % 256;  // Ensure it's in uint8 range
        }
        
        // Quantize the input tensor
        torch::Tensor quantized_tensor;
        try {
            // Convert to float first to ensure compatibility with quantization
            torch::Tensor float_tensor = input_tensor.to(torch::kFloat);
            
            // Quantize the tensor
            quantized_tensor = torch::quantize_per_tensor(
                float_tensor, scale, zero_point, torch::kQUInt8);
        } catch (const std::exception& e) {
            // If quantization fails, create a simple quantized tensor
            auto options = torch::TensorOptions().dtype(torch::kQUInt8);
            std::vector<int64_t> shape = {1, 4};
            quantized_tensor = torch::quantize_per_tensor(
                torch::ones(shape), 1.0, 0, torch::kQUInt8);
        }
        
        // Apply the softmax operation using functional API
        torch::Tensor output = torch::softmax(quantized_tensor.dequantize(), dim);
        
        // Try with different dimensions if tensor has multiple dimensions
        if (input_tensor.dim() > 1) {
            for (int d = 0; d < input_tensor.dim(); d++) {
                torch::Tensor output_d = torch::softmax(quantized_tensor.dequantize(), d);
            }
        }
        
        // Try with different scales and zero points
        if (offset + 2*sizeof(double) <= Size) {
            double new_scale;
            int64_t new_zero_point;
            
            std::memcpy(&new_scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            new_scale = std::abs(new_scale);
            if (new_scale == 0.0) new_scale = 0.1;
            
            std::memcpy(&new_zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            new_zero_point = new_zero_point % 256;
            
            try {
                torch::Tensor float_tensor = input_tensor.to(torch::kFloat);
                torch::Tensor quantized_tensor2 = torch::quantize_per_tensor(
                    float_tensor, new_scale, new_zero_point, torch::kQUInt8);
                torch::Tensor output2 = torch::softmax(quantized_tensor2.dequantize(), dim);
            } catch (const std::exception&) {
                // Ignore exceptions from this additional test
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
