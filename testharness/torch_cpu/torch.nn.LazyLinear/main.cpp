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
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for Linear (using regular Linear instead of LazyLinear)
        int64_t in_features = 0;
        int64_t out_features = 0;
        bool bias = true;
        
        // Get in_features from the input tensor if possible
        if (input.dim() >= 1) {
            in_features = input.size(-1);
        } else {
            // For scalar or empty tensor, use a default value
            in_features = 1;
        }
        
        // Get out_features from the remaining data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make out_features reasonable but allow edge cases
            out_features = std::abs(out_features) % 1024 + 1;
        } else {
            out_features = 10; // Default value
        }
        
        // Get bias parameter if data available
        if (offset < Size) {
            bias = Data[offset++] & 0x1; // Use lowest bit to determine bias
        }
        
        // Create Linear module (using regular Linear instead of LazyLinear)
        torch::nn::Linear linear(torch::nn::LinearOptions(in_features, out_features).bias(bias));
        
        // Apply the Linear module to the input tensor
        torch::Tensor output;
        
        // Handle different input dimensions
        if (input.dim() == 0) {
            // For scalar input, reshape to 1D tensor with one element
            output = linear(input.reshape({1, 1}));
        } else if (input.dim() == 1) {
            // For 1D input, add batch dimension
            output = linear(input.unsqueeze(0));
        } else {
            // For 2D+ inputs, apply directly
            output = linear(input);
        }
        
        // Force computation to ensure any errors are triggered
        output = output.contiguous();
        
        // Access some elements to ensure computation
        if (output.numel() > 0) {
            float sum = output.sum().item<float>();
            (void)sum; // Prevent unused variable warning
        }
        
        // Test with different input shapes if possible
        if (offset + 4 < Size && input.dim() > 0) {
            // Create another tensor with different shape but same last dimension
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Ensure the last dimension matches in_features
            std::vector<int64_t> new_shape = input2.sizes().vec();
            if (!new_shape.empty()) {
                new_shape.back() = in_features;
                input2 = input2.reshape(new_shape);
                
                // Apply the same Linear module to the new input
                torch::Tensor output2 = linear(input2);
                output2 = output2.contiguous();
            }
        }
        
        // Test with zero batch size if possible
        if (input.dim() >= 2) {
            std::vector<int64_t> zero_batch_shape = input.sizes().vec();
            zero_batch_shape[0] = 0;
            
            try {
                torch::Tensor zero_batch_input = torch::empty(zero_batch_shape, input.options());
                torch::Tensor zero_batch_output = linear(zero_batch_input);
                zero_batch_output = zero_batch_output.contiguous();
            } catch (const std::exception&) {
                // Ignore exceptions for zero-batch case
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
