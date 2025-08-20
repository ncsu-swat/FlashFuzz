#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Parse input data for BNReLU3d parameters
        int64_t num_features = 0;
        float eps = 1e-5;
        float momentum = 0.1;
        
        // Extract num_features (must be positive)
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&num_features, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            num_features = std::abs(num_features) % 64 + 1; // Ensure positive and reasonable size
        } else {
            num_features = 3; // Default value
        }
        
        // Extract eps
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&eps, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure eps is positive and not too small
            if (std::isnan(eps) || std::isinf(eps) || eps <= 0) {
                eps = 1e-5;
            }
        }
        
        // Extract momentum
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&momentum, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure momentum is between 0 and 1
            if (std::isnan(momentum) || std::isinf(momentum) || momentum < 0 || momentum > 1) {
                momentum = 0.1;
            }
        }
        
        // Create the BatchNorm3d module (since BNReLU3d is not available)
        torch::nn::BatchNorm3d bn3d(torch::nn::BatchNorm3dOptions(num_features)
                                    .eps(eps)
                                    .momentum(momentum));
        
        // Create input tensor
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure the input tensor has the right shape for BatchNorm3d (N, C, D, H, W)
            // where C should match num_features
            if (input.dim() != 5) {
                // Reshape to 5D if not already
                std::vector<int64_t> new_shape;
                if (input.dim() < 5) {
                    // Add dimensions as needed
                    new_shape = input.sizes().vec();
                    while (new_shape.size() < 5) {
                        new_shape.insert(new_shape.begin(), 1);
                    }
                } else if (input.dim() > 5) {
                    // Collapse extra dimensions
                    new_shape.push_back(input.size(0)); // N
                    new_shape.push_back(num_features);  // C
                    
                    int64_t remaining_elements = 1;
                    for (int i = 2; i < input.dim(); i++) {
                        remaining_elements *= input.size(i);
                    }
                    
                    // Distribute remaining elements to D, H, W dimensions
                    int64_t d = std::max<int64_t>(1, static_cast<int64_t>(std::cbrt(remaining_elements)));
                    int64_t hw = remaining_elements / d;
                    int64_t h = std::max<int64_t>(1, static_cast<int64_t>(std::sqrt(hw)));
                    int64_t w = std::max<int64_t>(1, hw / h);
                    
                    new_shape.push_back(d);
                    new_shape.push_back(h);
                    new_shape.push_back(w);
                }
                
                // Reshape the tensor
                input = input.reshape(new_shape);
            }
            
            // Ensure channel dimension matches num_features
            if (input.size(1) != num_features) {
                std::vector<int64_t> new_shape = input.sizes().vec();
                new_shape[1] = num_features;
                input = input.reshape(new_shape);
            }
            
            // Apply BatchNorm3d
            auto bn_output = bn3d(input.to(torch::kFloat));
            
            // Apply ReLU
            auto output = torch::relu(bn_output);
            
            // Quantize the output tensor
            auto scale = 1.0f / 256.0f;
            auto zero_point = 0;
            
            // Quantize the output tensor to uint8
            auto q_output = torch::quantize_per_tensor(
                output,
                scale,
                zero_point,
                torch::kQUInt8
            );
            
            // Dequantize for verification
            auto dq_output = q_output.dequantize();
            
            // Verify output has same shape as input
            if (dq_output.sizes() != input.sizes()) {
                throw std::runtime_error("Output shape doesn't match input shape");
            }
            
        } catch (const std::exception& e) {
            // Catch exceptions from tensor creation or module application
            return 0;
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}