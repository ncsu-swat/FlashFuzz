#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
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
        size_t offset = 0;
        
        if (Size < 8) {
            return 0;
        }
        
        // Get initialization type first
        uint8_t init_type = Data[offset++];
        
        // Get float parameters
        float param1 = 0.01f;
        float param2 = 1.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&param1, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isnan(param1) || std::isinf(param1)) {
                param1 = 0.01f;
            }
            // Clamp to reasonable range
            param1 = std::max(-10.0f, std::min(10.0f, param1));
        }
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&param2, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isnan(param2) || std::isinf(param2)) {
                param2 = 1.0f;
            }
            param2 = std::max(-10.0f, std::min(10.0f, param2));
        }
        
        // Create tensors with specific dimensions for different init methods
        // Most init methods require float tensors
        int64_t dim1 = 4 + (Data[0] % 8);  // 4-11
        int64_t dim2 = 4 + (Data[1] % 8);  // 4-11
        int64_t dim3 = 3 + (Data[2] % 4);  // 3-6
        
        switch (init_type % 12) {
            case 0: {
                // Xavier uniform - requires at least 2D
                auto tensor = torch::empty({dim1, dim2}, torch::kFloat);
                double gain = std::abs(param1) + 0.1;
                torch::nn::init::xavier_uniform_(tensor, gain);
                break;
            }
            case 1: {
                // Xavier normal - requires at least 2D
                auto tensor = torch::empty({dim1, dim2}, torch::kFloat);
                double gain = std::abs(param1) + 0.1;
                torch::nn::init::xavier_normal_(tensor, gain);
                break;
            }
            case 2: {
                // Kaiming uniform - requires at least 2D
                auto tensor = torch::empty({dim1, dim2}, torch::kFloat);
                double a = param1;  // negative slope for leaky relu
                torch::nn::init::kaiming_uniform_(tensor, a);
                break;
            }
            case 3: {
                // Kaiming normal - requires at least 2D
                auto tensor = torch::empty({dim1, dim2}, torch::kFloat);
                double a = param1;
                torch::nn::init::kaiming_normal_(tensor, a);
                break;
            }
            case 4: {
                // Uniform initialization
                auto tensor = torch::empty({dim1, dim2}, torch::kFloat);
                double low = std::min((double)param1, (double)param2);
                double high = std::max((double)param1, (double)param2);
                if (low == high) high = low + 1.0;
                torch::nn::init::uniform_(tensor, low, high);
                break;
            }
            case 5: {
                // Normal initialization
                auto tensor = torch::empty({dim1, dim2}, torch::kFloat);
                double mean = param1;
                double std = std::abs(param2) + 0.01;
                torch::nn::init::normal_(tensor, mean, std);
                break;
            }
            case 6: {
                // Constant initialization
                auto tensor = torch::empty({dim1, dim2}, torch::kFloat);
                torch::nn::init::constant_(tensor, param1);
                break;
            }
            case 7: {
                // Ones initialization
                auto tensor = torch::empty({dim1, dim2}, torch::kFloat);
                torch::nn::init::ones_(tensor);
                break;
            }
            case 8: {
                // Zeros initialization
                auto tensor = torch::empty({dim1, dim2}, torch::kFloat);
                torch::nn::init::zeros_(tensor);
                break;
            }
            case 9: {
                // Eye initialization - only works for 2D tensors
                auto tensor = torch::empty({dim1, dim2}, torch::kFloat);
                torch::nn::init::eye_(tensor);
                break;
            }
            case 10: {
                // Dirac initialization - requires 3, 4, or 5D tensors
                // For conv1d (3D), conv2d (4D), conv3d (5D)
                auto tensor = torch::empty({dim1, dim2, dim3}, torch::kFloat);
                try {
                    torch::nn::init::dirac_(tensor);
                } catch (...) {
                    // Dirac has specific requirements, silently ignore failures
                }
                break;
            }
            case 11: {
                // Orthogonal initialization - requires at least 2D
                auto tensor = torch::empty({dim1, dim2}, torch::kFloat);
                double gain = std::abs(param1) + 0.1;
                torch::nn::init::orthogonal_(tensor, gain);
                break;
            }
        }
        
        // Also test with tensors created from fuzzer data
        if (Size > offset + 16) {
            torch::Tensor fuzz_tensor = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Only apply simple inits that work with any tensor
            if (fuzz_tensor.is_floating_point() && fuzz_tensor.numel() > 0) {
                uint8_t simple_init = Data[0] % 4;
                try {
                    switch (simple_init) {
                        case 0:
                            torch::nn::init::zeros_(fuzz_tensor);
                            break;
                        case 1:
                            torch::nn::init::ones_(fuzz_tensor);
                            break;
                        case 2:
                            torch::nn::init::constant_(fuzz_tensor, param1);
                            break;
                        case 3:
                            torch::nn::init::uniform_(fuzz_tensor);
                            break;
                    }
                } catch (...) {
                    // Silently ignore - these are expected to fail sometimes
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}