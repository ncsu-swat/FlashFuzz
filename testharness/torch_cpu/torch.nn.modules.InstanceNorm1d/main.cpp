#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful input
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for InstanceNorm1d from the remaining data
        bool affine = false;
        bool track_running_stats = false;
        double eps = 1e-5;
        double momentum = 0.1;
        
        if (offset + 2 <= Size) {
            affine = Data[offset++] & 0x1;
            track_running_stats = Data[offset++] & 0x1;
        }
        
        if (offset + sizeof(double) <= Size) {
            eps = *reinterpret_cast<const double*>(Data + offset);
            offset += sizeof(double);
            // Ensure eps is positive
            eps = std::abs(eps);
            if (eps == 0) eps = 1e-5;
        }
        
        if (offset + sizeof(double) <= Size) {
            momentum = *reinterpret_cast<const double*>(Data + offset);
            offset += sizeof(double);
            // Ensure momentum is between 0 and 1
            momentum = std::abs(momentum);
            if (momentum > 1.0) momentum = momentum - std::floor(momentum);
        }
        
        // Get the number of features (channels) from the input tensor
        int64_t num_features = 1;
        
        // InstanceNorm1d expects input of shape [N, C, L]
        // If tensor has fewer dimensions, we'll reshape it
        if (input.dim() >= 2) {
            num_features = input.size(1);
        } else if (input.dim() == 1) {
            // For 1D tensor, reshape to [1, 1, L]
            input = input.reshape({1, 1, input.size(0)});
        } else if (input.dim() == 0) {
            // For 0D tensor (scalar), reshape to [1, 1, 1]
            input = input.reshape({1, 1, 1});
        }
        
        // Create InstanceNorm1d module
        torch::nn::InstanceNorm1d instance_norm(
            torch::nn::InstanceNorm1dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Apply InstanceNorm1d to the input tensor
        torch::Tensor output = instance_norm(input);
        
        // Perform some operations on the output to ensure it's used
        auto sum = output.sum();
        
        // Try different input shapes
        if (offset + 1 < Size) {
            uint8_t reshape_option = Data[offset++];
            
            if (reshape_option % 3 == 0 && input.dim() >= 3) {
                // Try with a different batch size
                auto shape = input.sizes().vec();
                shape[0] = (shape[0] > 1) ? shape[0] - 1 : shape[0] + 1;
                
                try {
                    torch::Tensor reshaped = input.reshape(shape);
                    output = instance_norm(reshaped);
                } catch (const std::exception&) {
                    // Reshape might fail, that's okay
                }
            } else if (reshape_option % 3 == 1) {
                // Try with a single sample
                try {
                    if (input.dim() >= 3) {
                        torch::Tensor single_sample = input.slice(0, 0, 1).squeeze(0);
                        output = instance_norm(single_sample.unsqueeze(0));
                    }
                } catch (const std::exception&) {
                    // Operation might fail, that's okay
                }
            } else {
                // Try with a different length
                try {
                    if (input.dim() >= 3) {
                        auto shape = input.sizes().vec();
                        shape[2] = (shape[2] > 1) ? shape[2] - 1 : shape[2] + 1;
                        torch::Tensor reshaped = input.reshape(shape);
                        output = instance_norm(reshaped);
                    }
                } catch (const std::exception&) {
                    // Reshape might fail, that's okay
                }
            }
        }
        
        // Test with different data types
        if (offset < Size) {
            uint8_t dtype_option = Data[offset++];
            
            try {
                torch::ScalarType target_dtype = fuzzer_utils::parseDataType(dtype_option);
                torch::Tensor converted = input.to(target_dtype);
                output = instance_norm(converted);
            } catch (const std::exception&) {
                // Conversion might fail, that's okay
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