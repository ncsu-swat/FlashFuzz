#include "fuzzer_utils.h"
#include <iostream>
#include <algorithm>
#include <cstring>

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
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Read control bytes for dimensions and split configuration
        uint8_t dim0 = (Data[offset++] % 4) + 1;  // 1-4
        uint8_t dim1 = (Data[offset++] % 4) + 1;  // 1-4
        uint8_t dim2 = (Data[offset++] % 8) + 1;  // 1-8 (depth dimension for dsplit)
        uint8_t split_mode = Data[offset++];      // Controls sections vs indices mode
        uint8_t sections_hint = Data[offset++];   // For sections mode
        uint8_t dtype_hint = Data[offset++];      // For dtype selection

        // Create a 3D tensor (dsplit requires at least 3 dimensions)
        std::vector<int64_t> shape = {dim0, dim1, dim2};
        
        // Select dtype based on fuzzer input
        torch::ScalarType dtype;
        switch (dtype_hint % 4) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kInt32; break;
            default: dtype = torch::kInt64; break;
        }

        torch::Tensor input_tensor = torch::rand(shape, torch::TensorOptions().dtype(torch::kFloat32));
        if (dtype != torch::kFloat32) {
            input_tensor = input_tensor.to(dtype);
        }

        std::vector<torch::Tensor> result;

        if (split_mode % 2 == 0) {
            // Use sections variant
            // sections must evenly divide dim2
            int64_t sections = (sections_hint % dim2) + 1;
            // Find a valid divisor
            while (dim2 % sections != 0 && sections > 1) {
                sections--;
            }
            
            result = torch::dsplit(input_tensor, sections);
        } else {
            // Use indices variant
            // Create valid split indices within bounds
            std::vector<int64_t> indices;
            
            int num_splits = (sections_hint % 3) + 1;  // 1-3 split points
            
            for (int i = 0; i < num_splits && offset < Size; i++) {
                int64_t idx = Data[offset++] % dim2;
                if (idx > 0 && idx < dim2) {
                    indices.push_back(idx);
                }
            }
            
            if (!indices.empty()) {
                // Sort and remove duplicates
                std::sort(indices.begin(), indices.end());
                indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
                
                // Remove any index at 0 or equal to dim2 (would create empty tensors)
                indices.erase(
                    std::remove_if(indices.begin(), indices.end(), 
                        [dim2](int64_t x) { return x <= 0 || x >= dim2; }),
                    indices.end()
                );
            }
            
            if (!indices.empty()) {
                result = torch::dsplit(input_tensor, indices);
            } else {
                // Fallback: split into sections=1 (returns the whole tensor)
                result = torch::dsplit(input_tensor, 1);
            }
        }

        // Verify the result
        if (!result.empty()) {
            auto first_tensor = result[0];
            
            // Force evaluation
            if (first_tensor.numel() > 0) {
                volatile double sum = first_tensor.sum().item<double>();
                (void)sum;
            }
            
            // Also check last tensor if multiple results
            if (result.size() > 1) {
                auto last_tensor = result.back();
                if (last_tensor.numel() > 0) {
                    volatile double sum = last_tensor.sum().item<double>();
                    (void)sum;
                }
            }
        }

        // Test with 4D tensor as well for better coverage
        if (offset + 1 < Size && Data[offset] % 3 == 0) {
            uint8_t dim3 = (Data[offset++] % 4) + 1;
            std::vector<int64_t> shape4d = {dim0, dim1, dim2, dim3};
            torch::Tensor input_4d = torch::rand(shape4d);
            
            try {
                auto result_4d = torch::dsplit(input_4d, 1);
                if (!result_4d.empty() && result_4d[0].numel() > 0) {
                    volatile double sum = result_4d[0].sum().item<double>();
                    (void)sum;
                }
            } catch (...) {
                // Silently ignore - 4D test is optional
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