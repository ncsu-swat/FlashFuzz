#include "fuzzer_utils.h"
#include <iostream>
#include <vector>

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
        // Need at least some data
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Determine number of tensors to stack (2-4)
        uint8_t num_tensors = (Data[offset] % 3) + 2;
        offset++;
        
        // Determine the common shape for all tensors (except first dim can vary)
        // vstack requires: all tensors must have same number of dims (after reshaping 1D)
        // and all dims except dim 0 must match
        uint8_t num_cols = (Data[offset] % 8) + 1;  // 1-8 columns
        offset++;
        
        // Determine dtype
        torch::ScalarType dtype = torch::kFloat32;
        if (offset < Size) {
            uint8_t dtype_choice = Data[offset] % 4;
            offset++;
            switch (dtype_choice) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kInt32; break;
                case 3: dtype = torch::kInt64; break;
            }
        }
        
        std::vector<torch::Tensor> tensors;
        tensors.reserve(num_tensors);
        
        // Create tensors with compatible shapes
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            // Each tensor can have different number of rows (dim 0)
            uint8_t num_rows = (offset < Size) ? (Data[offset] % 5) + 1 : 1;
            offset++;
            
            try {
                // Create 2D tensor with shape [num_rows, num_cols]
                torch::Tensor tensor;
                if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
                    tensor = torch::randn({num_rows, num_cols}, torch::TensorOptions().dtype(dtype));
                } else {
                    tensor = torch::randint(0, 100, {num_rows, num_cols}, torch::TensorOptions().dtype(dtype));
                }
                tensors.push_back(tensor);
            } catch (...) {
                // Silently ignore tensor creation failures
            }
        }
        
        // Need at least one tensor
        if (tensors.empty()) {
            return 0;
        }
        
        // Test vstack with vector of tensors
        torch::Tensor result = torch::vstack(tensors);
        
        // Verify result shape
        int64_t expected_rows = 0;
        for (const auto& t : tensors) {
            expected_rows += t.size(0);
        }
        
        // Use result to prevent optimization
        volatile int64_t result_rows = result.size(0);
        volatile int64_t result_cols = result.size(1);
        volatile int64_t numel = result.numel();
        
        // Also test with 1D tensors (vstack should treat them as row vectors)
        if (offset + 2 < Size) {
            uint8_t vec_len = (Data[offset] % 6) + 1;
            offset++;
            uint8_t num_vecs = (Data[offset] % 3) + 2;
            offset++;
            
            std::vector<torch::Tensor> vec_tensors;
            for (uint8_t i = 0; i < num_vecs; ++i) {
                vec_tensors.push_back(torch::randn({vec_len}));
            }
            
            try {
                torch::Tensor vec_result = torch::vstack(vec_tensors);
                volatile int64_t vec_numel = vec_result.numel();
            } catch (...) {
                // Expected if shapes don't align
            }
        }
        
        // Test with single tensor
        if (!tensors.empty()) {
            try {
                torch::Tensor single_result = torch::vstack({tensors[0]});
                volatile bool same = single_result.equal(tensors[0].reshape({tensors[0].size(0), -1}));
            } catch (...) {
                // May fail for 0-d tensors
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