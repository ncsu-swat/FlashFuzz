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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 2D tensor for linear operation
        // If not, reshape it to make it compatible
        if (input_tensor.dim() < 2) {
            int64_t batch_size = input_tensor.numel() > 0 ? input_tensor.numel() : 1;
            int64_t in_features = 4; // Arbitrary small feature size
            input_tensor = input_tensor.reshape({batch_size, in_features});
        }
        
        // Extract dimensions for the linear layer
        int64_t batch_size = input_tensor.size(0);
        int64_t in_features = input_tensor.size(1);
        
        // Get some bytes for out_features if available
        int64_t out_features = 4; // Default
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Ensure out_features is positive and reasonable
            out_features = std::abs(out_features) % 32 + 1;
        }
        
        // Create a regular Linear layer followed by ReLU since intrinsic quantized modules
        // are not directly available in the C++ frontend
        torch::nn::Linear linear(in_features, out_features);
        
        // Convert input to float if needed
        if (input_tensor.scalar_type() != torch::kFloat) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Apply linear transformation followed by ReLU
        torch::Tensor linear_output = linear->forward(input_tensor);
        torch::Tensor output = torch::relu(linear_output);
        
        // Verify output has expected shape
        if (output.size(0) != batch_size || output.size(1) != out_features) {
            throw std::runtime_error("Output tensor has unexpected shape");
        }
        
        // Verify ReLU effect: all values should be non-negative
        if (torch::any(output < 0).item<bool>()) {
            throw std::runtime_error("Output contains negative values after LinearReLU");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
