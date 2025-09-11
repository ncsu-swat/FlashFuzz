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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for group_norm
        // We need at least 4 more bytes for parameters
        if (offset + 4 > Size) {
            return 0;
        }
        
        // Parse number of groups
        int64_t num_groups = 1;
        if (input.dim() > 1) {
            uint8_t groups_byte = Data[offset++];
            // Ensure num_groups is between 1 and the number of channels
            // For group_norm, the channel dimension is typically dim 1 for NCHW format
            int64_t num_channels = input.dim() > 1 ? input.size(1) : 1;
            if (num_channels > 0) {
                num_groups = (groups_byte % num_channels) + 1;
                // Ensure num_channels is divisible by num_groups
                while (num_channels % num_groups != 0 && num_groups > 1) {
                    num_groups--;
                }
            }
        }
        
        // Parse epsilon value
        float epsilon = 1e-5; // Default value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&epsilon, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure epsilon is positive
            epsilon = std::abs(epsilon);
            // Avoid extremely small values that might cause numerical issues
            if (epsilon < 1e-10) {
                epsilon = 1e-10;
            }
        }
        
        // Create weight and bias tensors if there's enough data
        torch::Tensor weight;
        torch::Tensor bias;
        
        // For group_norm, weight and bias should have shape [C] where C is the number of channels
        if (input.dim() > 1) {
            int64_t num_channels = input.size(1);
            
            if (offset < Size) {
                // Create weight tensor
                auto options = torch::TensorOptions().dtype(input.dtype());
                if (num_channels > 0) {
                    std::vector<int64_t> weight_shape = {num_channels};
                    
                    // Parse weight data
                    std::vector<uint8_t> weight_data;
                    size_t dtype_size = c10::elementSize(input.scalar_type());
                    size_t bytes_needed = num_channels * dtype_size;
                    
                    if (offset + bytes_needed <= Size) {
                        weight_data.resize(bytes_needed);
                        std::memcpy(weight_data.data(), Data + offset, bytes_needed);
                        offset += bytes_needed;
                        weight = torch::from_blob(weight_data.data(), weight_shape, options).clone();
                    } else {
                        // Not enough data, create ones tensor
                        weight = torch::ones(weight_shape, options);
                    }
                    
                    // Create bias tensor with similar approach
                    std::vector<int64_t> bias_shape = {num_channels};
                    
                    if (offset + bytes_needed <= Size) {
                        std::vector<uint8_t> bias_data(bytes_needed);
                        std::memcpy(bias_data.data(), Data + offset, bytes_needed);
                        offset += bytes_needed;
                        bias = torch::from_blob(bias_data.data(), bias_shape, options).clone();
                    } else {
                        // Not enough data, create zeros tensor
                        bias = torch::zeros(bias_shape, options);
                    }
                }
            }
        }
        
        // Apply group_norm operation
        torch::Tensor output;
        
        // Handle different cases based on available parameters
        if (weight.defined() && bias.defined()) {
            output = torch::group_norm(input, num_groups, weight, bias, epsilon);
        } else {
            output = torch::group_norm(input, num_groups, {}, {}, epsilon);
        }
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        
        // Ensure the operation is not optimized away
        if (sum.item<float>() == -12345.6789f) {
            std::cerr << "Unlikely value detected";
        }
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
