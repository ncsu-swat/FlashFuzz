#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Read dimensions for input tensor (2D: variables x observations)
        uint8_t num_vars = (Data[offset++] % 8) + 1;      // 1-8 variables
        uint8_t num_obs = (Data[offset++] % 16) + 2;      // 2-17 observations (need at least 2)
        
        // Read correction parameter
        int64_t correction = Data[offset++] % 3;  // 0, 1, or 2
        
        // Read flags for optional weights
        uint8_t flags = Data[offset++];
        bool use_fweights = flags & 0x1;
        bool use_aweights = flags & 0x2;
        
        // Ensure we have enough data for tensor creation
        size_t tensor_bytes_needed = num_vars * num_obs * sizeof(float);
        if (offset + tensor_bytes_needed > Size) {
            return 0;
        }
        
        // Create 2D input tensor (variables x observations)
        std::vector<float> input_data(num_vars * num_obs);
        for (size_t i = 0; i < input_data.size() && offset < Size; i++) {
            // Create float values from bytes
            input_data[i] = static_cast<float>(static_cast<int8_t>(Data[offset++])) / 10.0f;
        }
        
        torch::Tensor input = torch::from_blob(
            input_data.data(), 
            {num_vars, num_obs}, 
            torch::kFloat32
        ).clone();  // Clone to own the data
        
        // Prepare optional weight tensors
        c10::optional<torch::Tensor> fweights_opt = c10::nullopt;
        c10::optional<torch::Tensor> aweights_opt = c10::nullopt;
        
        // Create fweights if requested (1D int tensor, size = num_obs)
        if (use_fweights && offset + num_obs <= Size) {
            std::vector<int64_t> fweights_data(num_obs);
            for (int i = 0; i < num_obs && offset < Size; i++) {
                // Positive integer weights
                fweights_data[i] = (Data[offset++] % 5) + 1;  // 1-5
            }
            fweights_opt = torch::from_blob(
                fweights_data.data(),
                {num_obs},
                torch::kInt64
            ).clone();
        }
        
        // Create aweights if requested (1D float tensor, size = num_obs)
        if (use_aweights && offset + num_obs <= Size) {
            std::vector<float> aweights_data(num_obs);
            for (int i = 0; i < num_obs && offset < Size; i++) {
                // Non-negative float weights
                aweights_data[i] = static_cast<float>(Data[offset++] % 100) / 10.0f + 0.1f;
            }
            aweights_opt = torch::from_blob(
                aweights_data.data(),
                {num_obs},
                torch::kFloat32
            ).clone();
        }
        
        // Call torch::cov with appropriate parameters
        torch::Tensor result;
        
        try {
            result = torch::cov(input, correction, fweights_opt, aweights_opt);
        } catch (const std::exception&) {
            // Shape mismatches or invalid weight values are expected
            return 0;
        }
        
        // Verify result is computed (should be num_vars x num_vars covariance matrix)
        if (result.defined()) {
            auto sizes = result.sizes();
            volatile int64_t numel = result.numel();
            
            // Access some elements to force computation
            if (numel > 0) {
                volatile float first = result.flatten()[0].item<float>();
                if (numel > 1) {
                    volatile float last = result.flatten()[-1].item<float>();
                }
            }
        }
        
        // Test with 1D input (single variable case)
        if (offset + 8 <= Size) {
            uint8_t obs_count = (Data[offset++] % 8) + 2;
            std::vector<float> vec_data(obs_count);
            for (int i = 0; i < obs_count && offset < Size; i++) {
                vec_data[i] = static_cast<float>(static_cast<int8_t>(Data[offset++])) / 10.0f;
            }
            
            torch::Tensor vec_input = torch::from_blob(
                vec_data.data(),
                {obs_count},
                torch::kFloat32
            ).clone();
            
            try {
                torch::Tensor vec_result = torch::cov(vec_input, correction);
                if (vec_result.defined() && vec_result.numel() == 1) {
                    volatile float val = vec_result.item<float>();
                }
            } catch (const std::exception&) {
                // Expected for some inputs
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