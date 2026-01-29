#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Get a byte to determine which initialization function to use
        uint8_t init_type = Data[offset++];
        
        // Get dimension info for creating appropriately shaped tensors
        uint8_t dim_selector = Data[offset++];
        
        // Get parameter values
        float param_value = 0.01f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&param_value, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isnan(param_value) || std::isinf(param_value) || std::abs(param_value) > 1e6f) {
                param_value = 0.01f;
            }
        }
        
        float param_value2 = 1.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&param_value2, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isnan(param_value2) || std::isinf(param_value2) || std::abs(param_value2) > 1e6f) {
                param_value2 = 1.0f;
            }
        }

        // Create tensors with appropriate shapes for different init functions
        // Use floating point type as required by init functions
        int init_func = init_type % 12;
        
        try {
            torch::Tensor tensor;
            
            // Create tensor with shape appropriate for the init function
            int d1 = (dim_selector % 8) + 2;  // 2-9
            int d2 = ((dim_selector >> 3) % 8) + 2;  // 2-9
            int d3 = ((dim_selector >> 6) % 4) + 2;  // 2-5
            
            switch (init_func) {
                case 0: // uniform_
                    tensor = torch::empty({d1, d2}, torch::kFloat32);
                    torch::nn::init::uniform_(tensor, -std::abs(param_value) - 0.001f, std::abs(param_value) + 0.001f);
                    break;
                    
                case 1: // normal_
                    tensor = torch::empty({d1, d2}, torch::kFloat32);
                    torch::nn::init::normal_(tensor, param_value, std::abs(param_value2) + 0.001f);
                    break;
                    
                case 2: // constant_
                    tensor = torch::empty({d1, d2}, torch::kFloat32);
                    torch::nn::init::constant_(tensor, param_value);
                    break;
                    
                case 3: // ones_
                    tensor = torch::empty({d1, d2}, torch::kFloat32);
                    torch::nn::init::ones_(tensor);
                    break;
                    
                case 4: // zeros_
                    tensor = torch::empty({d1, d2}, torch::kFloat32);
                    torch::nn::init::zeros_(tensor);
                    break;
                    
                case 5: // eye_ - requires 2D tensor
                    tensor = torch::empty({d1, d2}, torch::kFloat32);
                    torch::nn::init::eye_(tensor);
                    break;
                    
                case 6: // dirac_ - requires 3, 4, or 5D tensor
                    {
                        int groups = 1;
                        int channels = d1;
                        // Ensure channels is divisible by groups for dirac_
                        tensor = torch::empty({channels, channels / groups, d3, d3}, torch::kFloat32);
                        torch::nn::init::dirac_(tensor);
                    }
                    break;
                    
                case 7: // xavier_uniform_ - requires at least 2D
                    tensor = torch::empty({d1, d2}, torch::kFloat32);
                    torch::nn::init::xavier_uniform_(tensor, std::abs(param_value) + 0.001f);
                    break;
                    
                case 8: // xavier_normal_ - requires at least 2D
                    tensor = torch::empty({d1, d2}, torch::kFloat32);
                    torch::nn::init::xavier_normal_(tensor, std::abs(param_value) + 0.001f);
                    break;
                    
                case 9: // kaiming_uniform_ - requires at least 2D
                    tensor = torch::empty({d1, d2}, torch::kFloat32);
                    torch::nn::init::kaiming_uniform_(tensor, std::abs(param_value) + 0.001f,
                                                     torch::kFanIn, torch::kLeakyReLU);
                    break;
                    
                case 10: // kaiming_normal_ - requires at least 2D
                    tensor = torch::empty({d1, d2}, torch::kFloat32);
                    torch::nn::init::kaiming_normal_(tensor, std::abs(param_value) + 0.001f,
                                                    torch::kFanOut, torch::kReLU);
                    break;
                    
                case 11: // orthogonal_ - requires at least 2D
                    tensor = torch::empty({d1, d2}, torch::kFloat32);
                    torch::nn::init::orthogonal_(tensor, std::abs(param_value) + 0.001f);
                    break;
            }
            
            // Verify the initialization worked
            auto sum = tensor.sum();
            (void)sum;
            
        } catch (const c10::Error& e) {
            // Expected errors from shape mismatches, etc.
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}