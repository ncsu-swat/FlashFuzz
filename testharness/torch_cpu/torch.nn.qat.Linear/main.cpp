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
        
        // Need at least a few bytes for basic parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for Linear module
        int64_t in_features = 0;
        int64_t out_features = 0;
        bool bias = false;
        
        // Get in_features from the input tensor if possible
        if (input_tensor.dim() >= 2) {
            in_features = input_tensor.size(-1);
        } else if (input_tensor.dim() == 1) {
            in_features = input_tensor.size(0);
        } else {
            // For scalar tensors, use a small value
            in_features = 4;
        }
        
        // Get out_features from the remaining data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure out_features is reasonable
            out_features = std::abs(out_features) % 32 + 1;
        } else {
            out_features = 4;
        }
        
        // Determine if bias should be used
        if (offset < Size) {
            bias = Data[offset++] & 0x1;
        }
        
        // Create Linear module (QAT Linear is not available in PyTorch C++ API)
        torch::nn::Linear linear(
            torch::nn::LinearOptions(in_features, out_features).bias(bias)
        );
        
        // Set quantization parameters
        double scale = 1.0;
        int64_t zero_point = 0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure scale is positive and reasonable
            scale = std::abs(scale);
            if (scale < 1e-10) scale = 1e-10;
            if (scale > 1e10) scale = 1e10;
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure zero_point is within reasonable range
            zero_point = zero_point % 256;
        }
        
        // Reshape input tensor if needed to match expected input shape for Linear
        if (input_tensor.dim() == 0) {
            // For scalar tensors, reshape to 1D
            input_tensor = input_tensor.reshape({1});
        }
        
        if (input_tensor.dim() == 1) {
            // For 1D tensors, add batch dimension
            input_tensor = input_tensor.unsqueeze(0);
        }
        
        // If last dimension doesn't match in_features, reshape
        if (input_tensor.size(-1) != in_features) {
            std::vector<int64_t> new_shape = input_tensor.sizes().vec();
            if (!new_shape.empty()) {
                new_shape.back() = in_features;
                input_tensor = input_tensor.reshape(new_shape);
            }
        }
        
        // Apply the Linear module
        torch::Tensor output = linear->forward(input_tensor);
        
        // Test evaluation mode
        linear->train(false);
        torch::Tensor output_eval = linear->forward(input_tensor);
        
        // Test with different dtypes if possible
        if (input_tensor.dtype() != torch::kFloat) {
            torch::Tensor float_input = input_tensor.to(torch::kFloat);
            torch::Tensor float_output = linear->forward(float_input);
        }
        
        // Test with different device if available
        if (torch::cuda::is_available()) {
            torch::Tensor cuda_input = input_tensor.to(torch::kCUDA);
            torch::nn::Linear cuda_linear = linear->to(torch::kCUDA);
            torch::Tensor cuda_output = cuda_linear->forward(cuda_input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
