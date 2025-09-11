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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for histogram from remaining data
        int64_t bins = 10; // Default value
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&bins, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Ensure bins is within a reasonable range
            bins = std::abs(bins) % 1000 + 1; // Ensure at least 1 bin, max 1000
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
        
        // Ensure min_value <= max_value
        if (min_value > max_value) {
            std::swap(min_value, max_value);
        }
        
        // If they're equal, add a small offset to max_value
        if (min_value == max_value) {
            max_value += 1.0;
        }
        
        // Try different variants of histogram
        
        // Variant 1: Basic histogram with default parameters
        try {
            auto result1 = torch::histogram(input, bins);
        } catch (const std::exception& e) {
            // Just catch and continue
        }
        
        // Variant 2: Histogram with specified bins
        try {
            auto result2 = torch::histogram(input, bins);
        } catch (const std::exception& e) {
            // Just catch and continue
        }
        
        // Variant 3: Histogram with specified range
        try {
            std::vector<double> range_vec = {min_value, max_value};
            auto result3 = torch::histogram(
                input, 
                bins, 
                at::ArrayRef<double>(range_vec)
            );
        } catch (const std::exception& e) {
            // Just catch and continue
        }
        
        // Variant 4: Histogram with custom bin edges
        try {
            // Create bin edges tensor
            std::vector<double> edges;
            for (int i = 0; i <= bins; i++) {
                double edge = min_value + (max_value - min_value) * i / bins;
                edges.push_back(edge);
            }
            torch::Tensor bin_edges = torch::tensor(edges);
            auto result4 = torch::histogram(input, bin_edges);
        } catch (const std::exception& e) {
            // Just catch and continue
        }
        
        // Variant 5: Histogram with weight tensor
        try {
            // Create a weight tensor with the same shape as input
            torch::Tensor weights = torch::ones_like(input);
            std::vector<double> range_vec = {min_value, max_value};
            auto result5 = torch::histogram(
                input, 
                bins, 
                at::ArrayRef<double>(range_vec),
                weights
            );
        } catch (const std::exception& e) {
            // Just catch and continue
        }
        
        // Variant 6: Histogram with density normalization
        try {
            std::vector<double> range_vec = {min_value, max_value};
            auto result6 = torch::histogram(
                input, 
                bins, 
                at::ArrayRef<double>(range_vec),
                torch::nullopt,  // No weights
                true  // density=true
            );
        } catch (const std::exception& e) {
            // Just catch and continue
        }
        
        // Variant 7: Histogram with bin edges and weights
        try {
            std::vector<double> edges;
            for (int i = 0; i <= bins; i++) {
                double edge = min_value + (max_value - min_value) * i / bins;
                edges.push_back(edge);
            }
            torch::Tensor bin_edges = torch::tensor(edges);
            torch::Tensor weights = torch::ones_like(input);
            auto result7 = torch::histogram(input, bin_edges, weights);
        } catch (const std::exception& e) {
            // Just catch and continue
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
