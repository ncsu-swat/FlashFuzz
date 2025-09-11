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
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for Linear layer
        int64_t in_features = 0;
        int64_t out_features = 0;
        bool bias = true;
        
        // Get in_features from the input tensor if possible
        if (input_tensor.dim() >= 1) {
            in_features = input_tensor.size(-1);
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
            out_features = 4;  // Default value
        }
        
        // Get bias flag if data available
        if (offset < Size) {
            bias = Data[offset++] & 0x1;  // Use lowest bit
        }
        
        // Create the quantized dynamic linear layer
        auto linear = torch::nn::quantized::dynamic::Linear(in_features, out_features);
        if (!bias) {
            linear->options.bias(false);
        }
        
        // Reshape input tensor if needed to match expected input shape
        if (input_tensor.dim() == 0) {
            // Scalar tensor needs to be reshaped to have at least one dimension
            input_tensor = input_tensor.reshape({1, in_features});
        } else if (input_tensor.dim() == 1) {
            // 1D tensor needs to be reshaped to have batch dimension
            input_tensor = input_tensor.reshape({1, input_tensor.size(0)});
            
            // If the last dimension doesn't match in_features, reshape it
            if (input_tensor.size(-1) != in_features) {
                input_tensor = input_tensor.reshape({1, in_features});
            }
        } else {
            // For higher dimensional tensors, ensure the last dimension matches in_features
            auto sizes = input_tensor.sizes().vec();
            if (sizes.back() != in_features) {
                sizes.back() = in_features;
                input_tensor = input_tensor.reshape(sizes);
            }
        }
        
        // Apply the linear layer
        torch::Tensor output = linear->forward(input_tensor);
        
        // Try different input types
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Try with a different data type if possible
            try {
                torch::Tensor input_tensor2 = input_tensor.to(dtype);
                torch::Tensor output2 = linear->forward(input_tensor2);
            } catch (const std::exception&) {
                // Some data type conversions might not be supported, ignore
            }
        }
        
        // Try with empty tensor
        try {
            torch::Tensor empty_input = torch::empty({0, in_features});
            torch::Tensor empty_output = linear->forward(empty_input);
        } catch (const std::exception&) {
            // Empty tensor might not be supported, ignore
        }
        
        // Try with tensor containing extreme values
        try {
            torch::Tensor extreme_input = torch::full({1, in_features}, 1e10);
            torch::Tensor extreme_output = linear->forward(extreme_input);
        } catch (const std::exception&) {
            // Extreme values might cause issues, ignore
        }
        
        // Try with NaN values
        try {
            torch::Tensor nan_input = torch::full({1, in_features}, std::numeric_limits<float>::quiet_NaN());
            torch::Tensor nan_output = linear->forward(nan_input);
        } catch (const std::exception&) {
            // NaN values might cause issues, ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
