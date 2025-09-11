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
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for native_group_norm
        // We need: weight, bias, N, C, HxW, group, eps
        
        // Create weight and bias tensors if there's enough data left
        torch::Tensor weight;
        torch::Tensor bias;
        
        if (offset + 2 < Size) {
            // Try to create weight tensor
            weight = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try to create bias tensor if there's still data
            if (offset + 2 < Size) {
                bias = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                // Create empty bias tensor
                bias = torch::Tensor();
            }
        } else {
            // Create empty weight and bias tensors
            weight = torch::Tensor();
            bias = torch::Tensor();
        }
        
        // Extract N, C, HxW parameters from input tensor shape
        int64_t N = 1;
        int64_t C = 1;
        int64_t HxW = 1;
        
        if (input.dim() >= 3) {
            N = input.size(0);
            C = input.size(1);
            HxW = 1;
            for (int64_t i = 2; i < input.dim(); ++i) {
                HxW *= input.size(i);
            }
        } else if (input.dim() == 2) {
            N = input.size(0);
            C = input.size(1);
        } else if (input.dim() == 1) {
            C = input.size(0);
        }
        
        // Extract group and eps parameters from remaining data
        int64_t group = 1;
        double eps = 1e-5;
        
        if (offset + sizeof(int64_t) <= Size) {
            memcpy(&group, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure group is valid (between 1 and C)
            if (group <= 0) {
                group = 1;
            } else if (C > 0 && group > C) {
                group = C;
            }
        }
        
        if (offset + sizeof(double) <= Size) {
            memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure eps is positive
            if (eps <= 0 || !std::isfinite(eps)) {
                eps = 1e-5;
            }
        }
        
        // Call native_group_norm
        auto result = torch::native_group_norm(input, weight, bias, N, C, HxW, group, eps);
        
        // Access the result components to ensure they're computed
        auto output = std::get<0>(result);
        auto mean = std::get<1>(result);
        auto rstd = std::get<2>(result);
        
        // Force evaluation of the tensors
        output.sum().item<float>();
        if (mean.defined()) mean.sum().item<float>();
        if (rstd.defined()) rstd.sum().item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
