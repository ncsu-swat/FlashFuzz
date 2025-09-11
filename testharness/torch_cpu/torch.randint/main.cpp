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
        
        // Need at least 4 bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Parse low and high values for randint
        int64_t low = 0;
        int64_t high = 0;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&low, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&high, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Ensure high > low (randint requires this)
        if (high <= low) {
            std::swap(high, low);
            high = low + 1; // Ensure they're different
        }
        
        // Create output tensor shape
        torch::Tensor shape_tensor;
        if (offset < Size) {
            shape_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Try different variants of randint
        
        // Variant 1: Basic randint with scalar shape
        try {
            auto result1 = torch::randint(low, high, {});
        } catch (...) {}
        
        // Variant 2: randint with explicit shape
        try {
            auto result2 = torch::randint(low, high, {3, 4});
        } catch (...) {}
        
        // Variant 3: randint with shape from tensor
        if (shape_tensor.defined()) {
            try {
                std::vector<int64_t> shape_vec;
                for (int i = 0; i < shape_tensor.numel() && i < 8; i++) {
                    int64_t dim = std::abs(shape_tensor.data_ptr<float>()[i]) + 1;
                    if (dim > 1000) dim = 1000; // Limit dimension size
                    shape_vec.push_back(dim);
                }
                
                if (!shape_vec.empty()) {
                    auto result3 = torch::randint(low, high, shape_vec);
                }
            } catch (...) {}
        }
        
        // Variant 4: randint with options
        try {
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            auto result4 = torch::randint(low, high, {2, 3}, options);
        } catch (...) {}
        
        try {
            auto options = torch::TensorOptions().dtype(torch::kInt64);
            auto result5 = torch::randint(low, high, {5}, options);
        } catch (...) {}
        
        // Variant 5: Edge cases
        try {
            // Empty shape
            auto result6 = torch::randint(low, high, {0});
        } catch (...) {}
        
        try {
            // Large shape
            auto result7 = torch::randint(low, high, {1000, 1});
        } catch (...) {}
        
        try {
            // Negative bounds (should be handled by the swap above)
            auto result8 = torch::randint(low, high, {2, 2});
        } catch (...) {}
        
        try {
            // Different dtypes
            auto options1 = torch::TensorOptions().dtype(torch::kInt8);
            auto result9 = torch::randint(low, high, {3}, options1);
            
            auto options2 = torch::TensorOptions().dtype(torch::kInt16);
            auto result10 = torch::randint(low, high, {3}, options2);
            
            auto options3 = torch::TensorOptions().dtype(torch::kInt32);
            auto result11 = torch::randint(low, high, {3}, options3);
            
            auto options4 = torch::TensorOptions().dtype(torch::kFloat16);
            auto result12 = torch::randint(low, high, {3}, options4);
        } catch (...) {}
        
        // Variant 6: Single value randint
        try {
            auto result13 = torch::randint(high, {2, 2});
        } catch (...) {}
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
