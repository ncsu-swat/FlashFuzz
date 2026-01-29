#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Create mean tensor
        torch::Tensor mean = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create std tensor with remaining data
        torch::Tensor std_tensor;
        if (offset < Size) {
            std_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for a second tensor, create a scalar
            std_tensor = torch::tensor(1.0);
        }
        
        // Ensure std is positive (required for normal distribution)
        std_tensor = torch::abs(std_tensor) + 0.001f;
        
        // Variant 1: normal with mean tensor and std tensor
        if (mean.defined() && std_tensor.defined()) {
            try {
                torch::Tensor result1 = torch::normal(mean, std_tensor, c10::nullopt);
                (void)result1;
            } catch (...) {
                // Shape mismatch or other expected errors
            }
        }
        
        // Variant 2: normal with mean tensor and std scalar
        if (mean.defined() && offset + sizeof(float) <= Size) {
            float std_val;
            std::memcpy(&std_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Make std positive
            std_val = std::fabs(std_val) + 0.001f;
            
            try {
                torch::Tensor result2 = torch::normal(mean, static_cast<double>(std_val), c10::nullopt);
                (void)result2;
            } catch (...) {
                // Expected errors
            }
        }
        
        // Variant 3: normal with mean scalar and std tensor
        if (std_tensor.defined() && offset + sizeof(float) <= Size) {
            float mean_val;
            std::memcpy(&mean_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure std is positive
            torch::Tensor pos_std = torch::abs(std_tensor) + 0.001f;
            
            try {
                torch::Tensor result3 = torch::normal(static_cast<double>(mean_val), pos_std, c10::nullopt);
                (void)result3;
            } catch (...) {
                // Expected errors
            }
        }
        
        // Variant 4: normal with mean scalar, std scalar, and size
        if (offset + 2 * sizeof(float) <= Size) {
            float mean_val, std_val;
            std::memcpy(&mean_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            std::memcpy(&std_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Make std positive
            std_val = std::fabs(std_val) + 0.001f;
            
            // Create a reasonable size vector
            std::vector<int64_t> size_vec;
            if (offset < Size) {
                uint8_t dims = (Data[offset++] % 4) + 1; // 1 to 4 dimensions
                for (uint8_t i = 0; i < dims && offset < Size; i++) {
                    // Use small positive sizes to avoid memory issues
                    int64_t dim_size = (Data[offset++] % 16) + 1; // 1 to 16
                    size_vec.push_back(dim_size);
                }
            }
            
            if (size_vec.empty()) {
                size_vec = {4, 4}; // Default size
            }
            
            try {
                torch::Tensor result4 = torch::normal(
                    static_cast<double>(mean_val), 
                    static_cast<double>(std_val), 
                    size_vec);
                (void)result4;
            } catch (...) {
                // Expected errors
            }
        }
        
        // Variant 5: normal with generator
        if (offset + 2 * sizeof(float) + sizeof(uint64_t) <= Size) {
            float mean_val, std_val;
            std::memcpy(&mean_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            std::memcpy(&std_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Make std positive
            std_val = std::fabs(std_val) + 0.001f;
            
            uint64_t seed;
            std::memcpy(&seed, Data + offset, sizeof(uint64_t));
            offset += sizeof(uint64_t);
            
            try {
                auto gen = torch::Generator();
                gen.set_current_seed(seed);
                
                torch::Tensor result5 = torch::normal(
                    static_cast<double>(mean_val), 
                    static_cast<double>(std_val), 
                    {3, 3}, 
                    gen);
                (void)result5;
            } catch (...) {
                // Expected errors
            }
        }
        
        // Variant 6: normal_out with tensor inputs
        if (mean.defined() && std_tensor.defined()) {
            try {
                // Ensure std is positive
                torch::Tensor pos_std = torch::abs(std_tensor) + 0.001f;
                
                // Determine output shape via broadcasting
                auto out_sizes = mean.sizes().vec();
                torch::Tensor out = torch::empty(out_sizes, mean.options());
                torch::normal_out(out, mean, pos_std, c10::nullopt);
                (void)out;
            } catch (...) {
                // Shape mismatch or other expected errors
            }
        }
        
        // Variant 7: normal_out with scalar std (use 4-arg version to avoid ambiguity)
        if (mean.defined() && offset + sizeof(float) <= Size) {
            float std_val;
            std::memcpy(&std_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Make std positive
            std_val = std::fabs(std_val) + 0.001f;
            
            try {
                torch::Tensor out = torch::empty_like(mean);
                // Use the overload: normal_out(out, mean_tensor, std_double, generator)
                torch::normal_out(out, mean, static_cast<double>(std_val), c10::nullopt);
                (void)out;
            } catch (...) {
                // Expected errors
            }
        }
        
        // Variant 8: normal_out with scalar mean
        if (std_tensor.defined() && offset + sizeof(float) <= Size) {
            float mean_val;
            std::memcpy(&mean_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure std is positive
            torch::Tensor pos_std = torch::abs(std_tensor) + 0.001f;
            
            try {
                torch::Tensor out = torch::empty_like(pos_std);
                // Use the overload: normal_out(out, mean_double, std_tensor, generator)
                torch::normal_out(out, static_cast<double>(mean_val), pos_std, c10::nullopt);
                (void)out;
            } catch (...) {
                // Expected errors
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