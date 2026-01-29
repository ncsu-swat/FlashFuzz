#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

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
        
        // Need at least some data to derive dimensions
        if (Size < 8) {
            return 0;
        }
        
        // Extract dimensions from fuzzer data
        int64_t batch_size = (Data[offset++] % 16) + 1;      // 1-16
        int64_t in_features = (Data[offset++] % 32) + 1;     // 1-32
        int64_t out_features = (Data[offset++] % 32) + 1;    // 1-32
        bool bias_defined = (Data[offset++] % 2 == 0);
        
        // Create tensors with proper shapes for linear operation
        // grad_output: [batch, out_features]
        // input: [batch, in_features]
        // weight: [out_features, in_features]
        
        torch::Tensor grad_output = torch::randn({batch_size, out_features}, 
            torch::TensorOptions().dtype(torch::kFloat32));
        torch::Tensor input = torch::randn({batch_size, in_features}, 
            torch::TensorOptions().dtype(torch::kFloat32));
        torch::Tensor weight = torch::randn({out_features, in_features}, 
            torch::TensorOptions().dtype(torch::kFloat32));
        
        // Use remaining data to modify tensor values if available
        if (offset < Size) {
            float scale = static_cast<float>(Data[offset++]) / 255.0f * 2.0f;
            grad_output = grad_output * scale;
        }
        if (offset < Size) {
            float scale = static_cast<float>(Data[offset++]) / 255.0f * 2.0f;
            input = input * scale;
        }

        try {
            // Convert tensors to MKLDNN format
            torch::Tensor grad_output_mkldnn = grad_output.to_mkldnn();
            torch::Tensor input_mkldnn = input.to_mkldnn();
            torch::Tensor weight_mkldnn = weight.to_mkldnn();
            
            // Call mkldnn_linear_backward_weights
            auto result = torch::mkldnn_linear_backward_weights(
                grad_output_mkldnn,
                input_mkldnn,
                weight_mkldnn,
                bias_defined
            );
            
            // Access the results to ensure they're computed
            auto grad_weight = std::get<0>(result);
            auto grad_bias = std::get<1>(result);
            
            // Convert back from MKLDNN and force evaluation
            torch::Tensor grad_weight_cpu = grad_weight.to_dense();
            auto sum_weight = grad_weight_cpu.sum().item<float>();
            
            if (bias_defined && grad_bias.defined()) {
                torch::Tensor grad_bias_cpu = grad_bias.to_dense();
                auto sum_bias = grad_bias_cpu.sum().item<float>();
                
                // Prevent compiler from optimizing away
                if (sum_weight == -999999.0f && sum_bias == -999999.0f) {
                    std::cerr << "Unreachable" << std::endl;
                }
            } else {
                // Prevent compiler from optimizing away
                if (sum_weight == -999999.0f) {
                    std::cerr << "Unreachable" << std::endl;
                }
            }
        } catch (const c10::Error& e) {
            // PyTorch specific errors (shape mismatch, MKLDNN not available, etc.) are expected
        } catch (const std::runtime_error& e) {
            // Runtime errors from MKLDNN operations are expected
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}