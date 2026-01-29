#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstring>        // For memcpy

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
        size_t offset = 0;
        
        // Need at least 4 bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Parse low and high values for randint - use smaller integers to avoid overflow
        int32_t low_val = 0;
        int32_t high_val = 0;
        
        if (offset + sizeof(int32_t) <= Size) {
            std::memcpy(&low_val, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
        }
        
        if (offset + sizeof(int32_t) <= Size) {
            std::memcpy(&high_val, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
        }
        
        // Clamp to reasonable range to avoid overflow issues
        low_val = low_val % 10000;
        high_val = high_val % 10000;
        
        // Ensure high > low (randint requires this)
        int64_t low = static_cast<int64_t>(std::min(low_val, high_val));
        int64_t high = static_cast<int64_t>(std::max(low_val, high_val));
        if (high <= low) {
            high = low + 1;
        }
        
        // Parse shape dimensions from remaining data
        std::vector<int64_t> fuzz_shape;
        while (offset + 1 <= Size && fuzz_shape.size() < 4) {
            uint8_t dim_byte = Data[offset++];
            int64_t dim = (dim_byte % 10) + 1; // Dimensions 1-10
            fuzz_shape.push_back(dim);
        }
        if (fuzz_shape.empty()) {
            fuzz_shape.push_back(3);
            fuzz_shape.push_back(4);
        }
        
        // Variant 1: Basic randint with scalar shape (0-d tensor)
        try {
            auto result1 = torch::randint(low, high, {});
        } catch (...) {}
        
        // Variant 2: randint with explicit shape
        try {
            auto result2 = torch::randint(low, high, {3, 4});
        } catch (...) {}
        
        // Variant 3: randint with shape from fuzzer
        try {
            auto result3 = torch::randint(low, high, fuzz_shape);
        } catch (...) {}
        
        // Variant 4: randint with options - different dtypes
        try {
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            auto result4 = torch::randint(low, high, {2, 3}, options);
        } catch (...) {}
        
        try {
            auto options = torch::TensorOptions().dtype(torch::kInt64);
            auto result5 = torch::randint(low, high, {5}, options);
        } catch (...) {}
        
        try {
            auto options = torch::TensorOptions().dtype(torch::kInt32);
            auto result6 = torch::randint(low, high, {3, 3}, options);
        } catch (...) {}
        
        try {
            auto options = torch::TensorOptions().dtype(torch::kInt16);
            auto result7 = torch::randint(low, high, {4}, options);
        } catch (...) {}
        
        try {
            auto options = torch::TensorOptions().dtype(torch::kInt8);
            auto result8 = torch::randint(low, high, {2, 2}, options);
        } catch (...) {}
        
        // Variant 5: Single argument form (0 to high)
        try {
            int64_t single_high = std::abs(high) + 1;
            auto result9 = torch::randint(single_high, {2, 2});
        } catch (...) {}
        
        try {
            int64_t single_high = std::abs(high) + 1;
            auto result10 = torch::randint(single_high, fuzz_shape);
        } catch (...) {}
        
        // Variant 6: Edge cases with shapes
        try {
            // 1-D tensor
            auto result11 = torch::randint(low, high, {10});
        } catch (...) {}
        
        try {
            // Higher dimensional
            auto result12 = torch::randint(low, high, {2, 3, 4});
        } catch (...) {}
        
        try {
            // Large 1-D
            auto result13 = torch::randint(low, high, {1000});
        } catch (...) {}
        
        // Variant 7: randint_like (if a tensor exists)
        try {
            auto base_tensor = torch::zeros({3, 4}, torch::kInt64);
            auto result14 = torch::randint_like(base_tensor, low, high);
        } catch (...) {}
        
        try {
            auto base_tensor = torch::zeros(fuzz_shape, torch::kInt32);
            int64_t single_high = std::abs(high) + 1;
            auto result15 = torch::randint_like(base_tensor, single_high);
        } catch (...) {}
        
        // Variant 8: With generator (use default)
        try {
            auto gen = torch::make_generator<torch::CPUGeneratorImpl>();
            auto options = torch::TensorOptions().dtype(torch::kInt64);
            auto result16 = torch::randint(low, high, {4, 4}, gen, options);
        } catch (...) {}
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}