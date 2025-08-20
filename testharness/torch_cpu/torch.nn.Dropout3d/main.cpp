#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for tensor creation and dropout parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 4D tensor (batch_size, channels, height, width)
        // If not, reshape to make it compatible with Dropout3d
        if (input.dim() < 4) {
            std::vector<int64_t> new_shape;
            if (input.dim() == 0) {
                // Scalar tensor, reshape to [1, 1, 1, 1]
                new_shape = {1, 1, 1, 1};
            } else if (input.dim() == 1) {
                // 1D tensor, reshape to [1, input.size(0), 1, 1]
                new_shape = {1, input.size(0), 1, 1};
            } else if (input.dim() == 2) {
                // 2D tensor, reshape to [input.size(0), input.size(1), 1, 1]
                new_shape = {input.size(0), input.size(1), 1, 1};
            } else if (input.dim() == 3) {
                // 3D tensor, reshape to [input.size(0), input.size(1), input.size(2), 1]
                new_shape = {input.size(0), input.size(1), input.size(2), 1};
            }
            input = input.reshape(new_shape);
        }
        
        // Extract dropout probability from input data
        double p = 0.5; // Default value
        if (offset + sizeof(float) <= Size) {
            float p_raw;
            std::memcpy(&p_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Convert to a valid probability value [0, 1]
            p = std::abs(p_raw);
            p = p - std::floor(p); // Get fractional part
        }
        
        // Extract inplace flag from input data
        bool inplace = false;
        if (offset < Size) {
            inplace = (Data[offset++] & 0x01) != 0;
        }
        
        // Create Dropout3d module
        torch::nn::Dropout3d dropout(torch::nn::Dropout3dOptions().p(p).inplace(inplace));
        
        // Set training mode based on input data
        bool training = true;
        if (offset < Size) {
            training = (Data[offset++] & 0x01) != 0;
        }
        
        // Apply dropout in training mode
        dropout->train(training);
        
        // Apply dropout to input tensor
        torch::Tensor output = dropout(input);
        
        // Verify output has same shape as input
        if (output.sizes() != input.sizes()) {
            throw std::runtime_error("Output shape doesn't match input shape");
        }
        
        // Test edge case: zero probability
        torch::nn::Dropout3d dropout_zero(torch::nn::Dropout3dOptions().p(0.0).inplace(inplace));
        dropout_zero->train(training);
        torch::Tensor output_zero = dropout_zero(input);
        
        // Test edge case: probability 1.0 (drop everything)
        if (training) {
            torch::nn::Dropout3d dropout_one(torch::nn::Dropout3dOptions().p(1.0).inplace(inplace));
            dropout_one->train(true);
            torch::Tensor output_one = dropout_one(input);
        }
        
        // Test in eval mode (should be identity function)
        dropout->eval();
        torch::Tensor output_eval = dropout(input);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}