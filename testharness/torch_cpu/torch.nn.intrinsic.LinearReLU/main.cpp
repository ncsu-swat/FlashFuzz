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
        
        // Need at least a few bytes for meaningful fuzzing
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get input features dimension
        int64_t in_features = 1;
        if (input.dim() > 0) {
            in_features = input.size(-1);
        }
        
        // Get out_features from the remaining data
        int64_t out_features = 1;
        if (offset + sizeof(int64_t) <= Size) {
            int64_t raw_out_features;
            std::memcpy(&raw_out_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure out_features is positive but not too large
            out_features = std::abs(raw_out_features) % 128 + 1;
        }
        
        // Create Linear module and ReLU separately since intrinsic::LinearReLU doesn't exist
        torch::nn::Linear linear(in_features, out_features);
        
        // Set bias parameter if we have more data
        bool with_bias = true;
        if (offset < Size) {
            with_bias = Data[offset++] & 0x1;
            if (!with_bias) {
                linear->options.bias(false);
            }
        }
        
        // Reshape input if needed to match expected dimensions for linear layer
        if (input.dim() == 0) {
            // For scalar input, reshape to [1, in_features]
            input = input.reshape({1, in_features});
        } else if (input.dim() == 1) {
            // For 1D input, reshape to [1, in_features]
            if (input.size(0) != in_features) {
                input = input.reshape({1, in_features});
            }
        } else {
            // For N-D input, ensure last dimension is in_features
            std::vector<int64_t> new_shape = input.sizes().vec();
            if (new_shape.back() != in_features) {
                new_shape.back() = in_features;
                input = input.reshape(new_shape);
            }
        }
        
        // Apply Linear followed by ReLU (simulating LinearReLU)
        torch::Tensor linear_output = linear(input);
        torch::Tensor output = torch::relu(linear_output);
        
        // Verify output is not NaN or Inf
        if (output.isnan().any().item<bool>() || output.isinf().any().item<bool>()) {
            throw std::runtime_error("Output contains NaN or Inf values");
        }
        
        // Test with different data types if we have more data
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Try with the new data type if compatible with Linear
            if (dtype == torch::kFloat || dtype == torch::kDouble || 
                dtype == torch::kHalf || dtype == torch::kBFloat16) {
                torch::Tensor input_cast = input.to(dtype);
                torch::nn::Linear linear_cast(in_features, out_features);
                linear_cast->options.bias(with_bias);
                
                // Cast model parameters to the same dtype
                for (auto& param : linear_cast->parameters()) {
                    param.data() = param.data().to(dtype);
                }
                
                torch::Tensor linear_output_cast = linear_cast(input_cast);
                torch::Tensor output_cast = torch::relu(linear_output_cast);
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
