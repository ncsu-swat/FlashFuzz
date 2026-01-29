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
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor and flatten it (histogram expects 1D input)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float if not already floating point
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Flatten to 1D as histogram expects
        input = input.flatten();
        
        // Skip if empty
        if (input.numel() == 0) {
            return 0;
        }
        
        // Extract parameters for histogram from remaining data
        int64_t bins = 10;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&bins, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            bins = std::abs(bins) % 100 + 1; // 1 to 100 bins
        }
        
        // Extract range parameters if available
        double min_value = 0.0;
        double max_value = 1.0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&min_value, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&max_value, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Handle NaN/Inf values
        if (!std::isfinite(min_value)) {
            min_value = 0.0;
        }
        if (!std::isfinite(max_value)) {
            max_value = 1.0;
        }
        
        // Clamp to reasonable range
        min_value = std::max(-1e6, std::min(1e6, min_value));
        max_value = std::max(-1e6, std::min(1e6, max_value));
        
        // Ensure min_value < max_value
        if (min_value >= max_value) {
            max_value = min_value + 1.0;
        }
        
        // Variant 1: Basic histogram with number of bins
        try {
            auto result = torch::histogram(input, bins);
            // result is tuple<Tensor, Tensor> - hist and bin_edges
            torch::Tensor hist = std::get<0>(result);
            torch::Tensor bin_edges = std::get<1>(result);
            (void)hist;
            (void)bin_edges;
        } catch (...) {
            // Silently catch expected failures
        }
        
        // Variant 2: Histogram with specified range
        try {
            c10::optional<at::ArrayRef<double>> range = c10::nullopt;
            std::vector<double> range_vec = {min_value, max_value};
            range = at::ArrayRef<double>(range_vec);
            
            auto result = torch::histogram(input, bins, range);
            torch::Tensor hist = std::get<0>(result);
            torch::Tensor bin_edges = std::get<1>(result);
            (void)hist;
            (void)bin_edges;
        } catch (...) {
            // Silently catch expected failures
        }
        
        // Variant 3: Histogram with bin edges tensor
        try {
            std::vector<float> edges;
            for (int64_t i = 0; i <= bins; i++) {
                float edge = static_cast<float>(min_value + (max_value - min_value) * i / bins);
                edges.push_back(edge);
            }
            torch::Tensor bin_edges_tensor = torch::tensor(edges);
            
            auto result = torch::histogram(input, bin_edges_tensor);
            torch::Tensor hist = std::get<0>(result);
            torch::Tensor bin_edges = std::get<1>(result);
            (void)hist;
            (void)bin_edges;
        } catch (...) {
            // Silently catch expected failures
        }
        
        // Variant 4: Histogram with weight tensor
        try {
            torch::Tensor weights = torch::ones_like(input);
            std::vector<double> range_vec = {min_value, max_value};
            
            auto result = torch::histogram(
                input, 
                bins, 
                at::ArrayRef<double>(range_vec),
                weights
            );
            torch::Tensor hist = std::get<0>(result);
            torch::Tensor bin_edges = std::get<1>(result);
            (void)hist;
            (void)bin_edges;
        } catch (...) {
            // Silently catch expected failures
        }
        
        // Variant 5: Histogram with weight tensor and density=true
        try {
            torch::Tensor weights = torch::rand_like(input);
            std::vector<double> range_vec = {min_value, max_value};
            
            auto result = torch::histogram(
                input, 
                bins, 
                at::ArrayRef<double>(range_vec),
                weights,
                true  // density
            );
            torch::Tensor hist = std::get<0>(result);
            torch::Tensor bin_edges = std::get<1>(result);
            (void)hist;
            (void)bin_edges;
        } catch (...) {
            // Silently catch expected failures
        }
        
        // Variant 6: Histogram with bin edges tensor and weights
        try {
            std::vector<float> edges;
            for (int64_t i = 0; i <= bins; i++) {
                float edge = static_cast<float>(min_value + (max_value - min_value) * i / bins);
                edges.push_back(edge);
            }
            torch::Tensor bin_edges_tensor = torch::tensor(edges);
            torch::Tensor weights = torch::ones_like(input);
            
            auto result = torch::histogram(input, bin_edges_tensor, weights);
            torch::Tensor hist = std::get<0>(result);
            torch::Tensor bin_edges = std::get<1>(result);
            (void)hist;
            (void)bin_edges;
        } catch (...) {
            // Silently catch expected failures
        }
        
        // Variant 7: Test with double precision input
        try {
            torch::Tensor input_double = input.to(torch::kFloat64);
            auto result = torch::histogram(input_double, bins);
            torch::Tensor hist = std::get<0>(result);
            torch::Tensor bin_edges = std::get<1>(result);
            (void)hist;
            (void)bin_edges;
        } catch (...) {
            // Silently catch expected failures
        }
        
        // Variant 8: Histogram without weight, with density
        try {
            std::vector<double> range_vec = {min_value, max_value};
            
            auto result = torch::histogram(
                input, 
                bins, 
                at::ArrayRef<double>(range_vec),
                c10::nullopt,  // no weights
                true           // density
            );
            torch::Tensor hist = std::get<0>(result);
            torch::Tensor bin_edges = std::get<1>(result);
            (void)hist;
            (void)bin_edges;
        } catch (...) {
            // Silently catch expected failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}