#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create mean tensor
        torch::Tensor mean = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create std tensor with remaining data
        torch::Tensor std;
        if (offset < Size) {
            std = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for a second tensor, create a scalar
            std = torch::tensor(1.0);
        }
        
        // Test different variants of torch::normal
        
        // Variant 1: normal with mean tensor and std tensor
        if (mean.defined() && std.defined()) {
            try {
                torch::Tensor result1 = torch::normal(mean, std, torch::nullopt);
            } catch (...) {
                // Catch and continue to test other variants
            }
        }
        
        // Variant 2: normal with mean tensor and std scalar
        if (mean.defined() && offset + sizeof(float) <= Size) {
            float std_val;
            std::memcpy(&std_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            try {
                torch::Tensor result2 = torch::normal(mean, std_val, torch::nullopt);
            } catch (...) {
                // Catch and continue
            }
        }
        
        // Variant 3: normal with mean scalar and std tensor
        if (std.defined() && offset + sizeof(float) <= Size) {
            float mean_val;
            std::memcpy(&mean_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            try {
                torch::Tensor result3 = torch::normal(mean_val, std, torch::nullopt);
            } catch (...) {
                // Catch and continue
            }
        }
        
        // Variant 4: normal with mean scalar and std scalar
        if (offset + 2*sizeof(float) <= Size) {
            float mean_val, std_val;
            std::memcpy(&mean_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            std::memcpy(&std_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            try {
                torch::Tensor result4 = torch::normal(mean_val, std_val, mean.sizes());
            } catch (...) {
                // Catch and continue
            }
        }
        
        // Variant 5: normal with size parameter
        if (offset + 2*sizeof(float) <= Size) {
            float mean_val, std_val;
            std::memcpy(&mean_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            std::memcpy(&std_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Create a size vector from remaining bytes
            std::vector<int64_t> size_vec;
            if (offset < Size) {
                uint8_t dims = Data[offset++] % 5; // Up to 4 dimensions
                for (uint8_t i = 0; i < dims && offset + sizeof(int64_t) <= Size; i++) {
                    int64_t dim_size;
                    std::memcpy(&dim_size, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    // Allow any dimension size including negative to test error handling
                    size_vec.push_back(dim_size);
                }
            }
            
            try {
                if (!size_vec.empty()) {
                    torch::Tensor result5 = torch::normal(mean_val, std_val, size_vec);
                }
            } catch (...) {
                // Catch and continue
            }
        }
        
        // Variant 6: normal with generator
        if (offset + 2*sizeof(float) <= Size) {
            float mean_val, std_val;
            std::memcpy(&mean_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            std::memcpy(&std_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            try {
                auto gen = torch::make_generator<torch::CPUGeneratorImpl>();
                if (offset + sizeof(uint64_t) <= Size) {
                    uint64_t seed;
                    std::memcpy(&seed, Data + offset, sizeof(uint64_t));
                    gen.set_current_seed(seed);
                }
                
                torch::Tensor result6 = torch::normal(mean_val, std_val, {3, 3}, gen);
            } catch (...) {
                // Catch and continue
            }
        }
        
        // Variant 7: out variant
        if (mean.defined() && std.defined()) {
            try {
                torch::Tensor out = torch::empty_like(mean);
                torch::normal_out(out, mean, std);
            } catch (...) {
                // Catch and continue
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}