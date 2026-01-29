#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        // Need enough bytes for parameters
        if (Size < 16) {
            return 0;
        }
        
        // Extract dimensions from fuzzer data
        uint8_t N_byte = Data[offset++];
        uint8_t C_byte = Data[offset++];
        uint8_t H_byte = Data[offset++];
        uint8_t W_byte = Data[offset++];
        uint8_t group_byte = Data[offset++];
        
        // Constrain dimensions to reasonable values
        int64_t N = (N_byte % 4) + 1;        // 1-4
        int64_t C = (C_byte % 16) + 1;       // 1-16
        int64_t H = (H_byte % 8) + 1;        // 1-8
        int64_t W = (W_byte % 8) + 1;        // 1-8
        int64_t HxW = H * W;
        
        // Group must divide C evenly
        // Find valid group values: divisors of C
        std::vector<int64_t> valid_groups;
        for (int64_t g = 1; g <= C; ++g) {
            if (C % g == 0) {
                valid_groups.push_back(g);
            }
        }
        int64_t group = valid_groups[group_byte % valid_groups.size()];
        
        // Extract eps
        double eps = 1e-5;
        if (offset + sizeof(float) <= Size) {
            float eps_f;
            memcpy(&eps_f, Data + offset, sizeof(float));
            offset += sizeof(float);
            if (std::isfinite(eps_f) && eps_f > 0 && eps_f < 1.0f) {
                eps = static_cast<double>(eps_f);
            }
        }
        
        // Create input tensor with shape [N, C, H, W] or [N, C, HxW]
        torch::Tensor input = torch::randn({N, C, H, W}, torch::kFloat32);
        
        // Fill with fuzzer data if available
        if (offset < Size) {
            size_t tensor_size = std::min(Size - offset, static_cast<size_t>(N * C * H * W * sizeof(float)));
            auto input_data = input.data_ptr<float>();
            size_t num_elements = tensor_size / sizeof(float);
            for (size_t i = 0; i < num_elements && i < static_cast<size_t>(input.numel()); ++i) {
                memcpy(&input_data[i], Data + offset + i * sizeof(float), sizeof(float));
                if (!std::isfinite(input_data[i])) {
                    input_data[i] = 0.0f;
                }
            }
            offset += tensor_size;
        }
        
        // Determine whether to use weight/bias based on fuzzer data
        bool use_weight = (offset < Size) && (Data[offset++] % 2 == 0);
        bool use_bias = (offset < Size) && (Data[offset++] % 2 == 0);
        
        // Weight and bias must have shape [C]
        torch::Tensor weight;
        torch::Tensor bias;
        
        if (use_weight) {
            weight = torch::randn({C}, torch::kFloat32);
        }
        
        if (use_bias) {
            bias = torch::randn({C}, torch::kFloat32);
        }
        
        // Inner try-catch for expected shape/dtype errors
        try
        {
            // Call native_group_norm
            // Signature: native_group_norm(input, weight, bias, N, C, HxW, group, eps)
            auto result = torch::native_group_norm(input, weight, bias, N, C, HxW, group, eps);
            
            // Access the result components
            auto output = std::get<0>(result);
            auto mean = std::get<1>(result);
            auto rstd = std::get<2>(result);
            
            // Force evaluation
            if (output.defined() && output.numel() > 0) {
                output.sum().item<float>();
            }
            if (mean.defined() && mean.numel() > 0) {
                mean.sum().item<float>();
            }
            if (rstd.defined() && rstd.numel() > 0) {
                rstd.sum().item<float>();
            }
        }
        catch (const c10::Error &e)
        {
            // Expected errors from invalid shapes/parameters - ignore silently
        }
        catch (const std::runtime_error &e)
        {
            // Expected runtime errors - ignore silently
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}