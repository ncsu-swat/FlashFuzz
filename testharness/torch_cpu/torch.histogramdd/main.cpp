#include "fuzzer_utils.h"
#include <iostream>
#include <optional>
#include <cstring>
#include <algorithm>

// --- Fuzzer Entry Point ---
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
        // Need at least some data to work with
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Parse control bytes first
        uint8_t num_samples_byte = Data[offset++];
        uint8_t num_dims_byte = Data[offset++];
        uint8_t bins_byte = Data[offset++];
        uint8_t flags_byte = Data[offset++];
        
        // Limit dimensions to reasonable values to avoid OOM
        int64_t N = (num_samples_byte % 32) + 1;  // 1-32 samples
        int64_t D = (num_dims_byte % 4) + 1;      // 1-4 dimensions
        int64_t bins_per_dim = (bins_byte % 10) + 2; // 2-11 bins per dimension
        
        bool use_weight = (flags_byte & 0x01) != 0;
        bool use_density = (flags_byte & 0x02) != 0;
        bool use_range = (flags_byte & 0x04) != 0;
        
        // Calculate how much data we need for the input tensor
        size_t float_data_needed = N * D * sizeof(float);
        if (offset + float_data_needed > Size) {
            // Not enough data, use what we have
            int64_t available_floats = (Size - offset) / sizeof(float);
            if (available_floats < 2) {
                return 0;  // Need at least 2 floats
            }
            // Adjust N and D
            N = std::max(int64_t(1), available_floats / D);
            if (N * D > available_floats) {
                D = available_floats / N;
                if (D == 0) {
                    D = 1;
                    N = available_floats;
                }
            }
        }
        
        // Create input tensor from fuzzer data
        std::vector<float> input_data(N * D);
        size_t bytes_to_copy = std::min(input_data.size() * sizeof(float), Size - offset);
        std::memcpy(input_data.data(), Data + offset, bytes_to_copy);
        offset += bytes_to_copy;
        
        // Create float tensor with shape (N, D)
        torch::Tensor input = torch::from_blob(input_data.data(), {N, D}, torch::kFloat32).clone();
        
        // Ensure input is float and contiguous
        input = input.to(torch::kFloat64).contiguous();
        
        // Create bins array
        std::vector<int64_t> bins(D, bins_per_dim);
        
        // Create range if requested
        std::optional<std::vector<double>> range_opt;
        if (use_range && offset + D * 2 * sizeof(float) <= Size) {
            std::vector<double> range_flat;
            for (int64_t d = 0; d < D; d++) {
                float min_val, max_val;
                std::memcpy(&min_val, Data + offset, sizeof(float));
                offset += sizeof(float);
                std::memcpy(&max_val, Data + offset, sizeof(float));
                offset += sizeof(float);
                
                // Handle NaN/Inf
                if (!std::isfinite(min_val)) min_val = -10.0f;
                if (!std::isfinite(max_val)) max_val = 10.0f;
                
                // Ensure min < max
                if (min_val >= max_val) {
                    double temp = min_val;
                    min_val = max_val - 1.0f;
                    max_val = temp + 1.0f;
                    if (min_val >= max_val) {
                        min_val = -1.0;
                        max_val = 1.0;
                    }
                }
                
                range_flat.push_back(static_cast<double>(min_val));
                range_flat.push_back(static_cast<double>(max_val));
            }
            range_opt = range_flat;
        }
        
        // Create weight tensor if requested
        std::optional<torch::Tensor> weight_opt;
        if (use_weight) {
            // Weight must have shape matching input samples
            std::vector<float> weight_data(N);
            if (offset + N * sizeof(float) <= Size) {
                std::memcpy(weight_data.data(), Data + offset, N * sizeof(float));
                offset += N * sizeof(float);
            } else {
                // Fill with 1s
                std::fill(weight_data.begin(), weight_data.end(), 1.0f);
            }
            
            // Handle NaN/Inf in weights
            for (auto& w : weight_data) {
                if (!std::isfinite(w) || w < 0) {
                    w = 1.0f;
                }
            }
            
            torch::Tensor weight = torch::from_blob(weight_data.data(), {N}, torch::kFloat32).clone();
            weight = weight.to(torch::kFloat64);
            weight_opt = weight;
        }
        
        // Test 1: histogramdd with IntArrayRef bins
        try {
            auto result = at::histogramdd(
                input,
                at::IntArrayRef(bins),
                range_opt.has_value() 
                    ? std::optional<at::ArrayRef<double>>(at::ArrayRef<double>(range_opt.value()))
                    : std::nullopt,
                weight_opt,
                use_density
            );
            
            // Access results to ensure computation
            auto hist = std::get<0>(result);
            auto bin_edges = std::get<1>(result);
            
            // Verify histogram is valid
            if (hist.numel() > 0) {
                auto sum = hist.sum();
                (void)sum.item<double>();
            }
            
            // Verify bin_edges
            for (const auto& edge : bin_edges) {
                (void)edge.numel();
            }
        } catch (const c10::Error&) {
            // Expected PyTorch errors
        }
        
        // Test 2: histogramdd with single int64_t bins (uniform across all dims)
        try {
            auto result = at::histogramdd(
                input,
                bins_per_dim,
                std::nullopt,
                std::nullopt,
                false
            );
            
            auto hist = std::get<0>(result);
            auto bin_edges = std::get<1>(result);
            (void)hist.sum().item<double>();
        } catch (const c10::Error&) {
            // Expected PyTorch errors
        }
        
        // Test 3: histogramdd with TensorList bins
        try {
            std::vector<torch::Tensor> bin_tensors;
            for (int64_t d = 0; d < D; d++) {
                // Create linearly spaced bin edges
                torch::Tensor edges = torch::linspace(-10.0, 10.0, bins_per_dim + 1, torch::kFloat64);
                bin_tensors.push_back(edges);
            }
            
            auto result = at::histogramdd(
                input,
                at::TensorList(bin_tensors),
                std::nullopt,
                weight_opt,
                use_density
            );
            
            auto hist = std::get<0>(result);
            (void)hist.sum().item<double>();
        } catch (const c10::Error&) {
            // Expected PyTorch errors
        }
        
        // Test 4: Edge case - minimum samples
        try {
            torch::Tensor single_input = torch::randn({1, D}, torch::kFloat64);
            auto result = at::histogramdd(single_input, int64_t(3));
            (void)std::get<0>(result).numel();
        } catch (const c10::Error&) {
            // Expected PyTorch errors
        }
        
        // Test 5: Different data type (ensure float64 is used)
        try {
            torch::Tensor float32_input = input.to(torch::kFloat32);
            // histogramdd should still work but may convert internally
            auto result = at::histogramdd(float32_input, int64_t(5));
            (void)std::get<0>(result).numel();
        } catch (const c10::Error&) {
            // Expected PyTorch errors  
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}