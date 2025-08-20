#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 4 bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Parse scale and zero_point from input data
        float scale = 0.1f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure scale is positive
            scale = std::abs(scale);
            // Avoid extremely small or large scales
            if (scale < 1e-6f) scale = 1e-6f;
            if (scale > 1e6f) scale = 1e6f;
        }
        
        int64_t zero_point = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse quantization dtype
        torch::ScalarType dtype = torch::kQInt8;
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            // Choose between common quantization types
            if (dtype_selector % 3 == 0) {
                dtype = torch::kQInt8;
            } else if (dtype_selector % 3 == 1) {
                dtype = torch::kQUInt8;
            } else {
                dtype = torch::kQInt32;
            }
        }
        
        // Parse tensor shape
        uint8_t rank = 0;
        if (offset < Size) {
            rank = fuzzer_utils::parseRank(Data[offset++]);
        }
        
        std::vector<int64_t> shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
        
        // Create quantization parameters
        auto qscheme = at::kPerTensorAffine;
        if (offset < Size && Data[offset++] % 2 == 0) {
            qscheme = at::kPerTensorSymmetric;
        }
        
        // Create a quantized tensor to use as template
        try {
            // First create a regular tensor with the desired shape and dtype
            torch::Tensor temp_tensor = torch::zeros(shape, torch::TensorOptions().dtype(torch::kFloat));
            
            // Quantize it to create a template tensor
            torch::Tensor qtensor = torch::quantize_per_tensor(temp_tensor, scale, zero_point, dtype);
            
            // Create empty quantized tensor using the template
            torch::Tensor result = torch::empty_quantized(shape, qtensor);
            
            // Basic validation
            if (result.is_quantized()) {
                auto qparams = result.q_scale();
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected and part of testing
            return 0;
        }
        
        // Try with different device if available
        if (torch::cuda::is_available() && offset < Size && Data[offset++] % 2 == 0) {
            try {
                // Create template tensor on CUDA
                torch::Tensor temp_tensor = torch::zeros(shape, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
                torch::Tensor qtensor = torch::quantize_per_tensor(temp_tensor, scale, zero_point, dtype);
                
                auto options = torch::TensorOptions().device(torch::kCUDA);
                torch::Tensor result = torch::empty_quantized(shape, qtensor, options);
            } catch (const c10::Error& e) {
                // PyTorch-specific exceptions are expected
                return 0;
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