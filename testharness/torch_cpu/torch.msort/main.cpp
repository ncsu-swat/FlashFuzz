#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Skip if there's not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor from fuzzer data
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply msort operation - sorts along the first dimension
        torch::Tensor result = torch::msort(input);
        
        // Try with different tensor shapes driven by fuzzer data
        if (offset + 2 < Size) {
            int64_t rows = (Data[offset] % 8) + 1;  // 1-8 rows
            int64_t cols = (Data[offset + 1] % 8) + 1;  // 1-8 cols
            offset += 2;
            
            // Create a 2D tensor with deterministic values from fuzzer data
            torch::Tensor tensor_2d = torch::zeros({rows, cols});
            for (int64_t i = 0; i < rows && offset < Size; i++) {
                for (int64_t j = 0; j < cols && offset < Size; j++) {
                    tensor_2d[i][j] = static_cast<float>(Data[offset++]);
                }
            }
            
            torch::Tensor result_2d = torch::msort(tensor_2d);
        }
        
        // Try with 3D tensor
        if (offset + 3 < Size) {
            int64_t d0 = (Data[offset] % 4) + 1;
            int64_t d1 = (Data[offset + 1] % 4) + 1;
            int64_t d2 = (Data[offset + 2] % 4) + 1;
            offset += 3;
            
            torch::Tensor tensor_3d = torch::zeros({d0, d1, d2});
            torch::Tensor result_3d = torch::msort(tensor_3d);
        }
        
        // Try with empty tensor (edge case)
        try {
            torch::Tensor empty_tensor = torch::empty({0});
            torch::Tensor empty_result = torch::msort(empty_tensor);
        } catch (...) {
            // Empty tensor sorting may fail, that's expected
        }
        
        // Try with scalar tensor (0-dimensional)
        if (offset < Size) {
            torch::Tensor scalar_tensor = torch::tensor(static_cast<float>(Data[offset++]));
            try {
                torch::Tensor scalar_result = torch::msort(scalar_tensor);
            } catch (...) {
                // Scalar sorting may fail, that's expected
            }
        }
        
        // Try with different dtypes
        if (offset + 4 < Size) {
            uint8_t dtype_selector = Data[offset++] % 4;
            
            torch::Tensor typed_tensor;
            switch (dtype_selector) {
                case 0:
                    typed_tensor = torch::zeros({3, 4}, torch::kFloat32);
                    break;
                case 1:
                    typed_tensor = torch::zeros({3, 4}, torch::kFloat64);
                    break;
                case 2:
                    typed_tensor = torch::zeros({3, 4}, torch::kInt32);
                    break;
                case 3:
                    typed_tensor = torch::zeros({3, 4}, torch::kInt64);
                    break;
            }
            
            // Fill with fuzzer data
            auto accessor = typed_tensor.accessor<float, 2>();
            for (int64_t i = 0; i < 3 && offset < Size; i++) {
                for (int64_t j = 0; j < 4 && offset < Size; j++) {
                    typed_tensor[i][j] = static_cast<float>(Data[offset++]);
                }
            }
            
            torch::Tensor typed_result = torch::msort(typed_tensor);
        }
        
        // Try with contiguous and non-contiguous tensors
        if (offset + 2 < Size) {
            torch::Tensor base = torch::zeros({4, 4});
            for (int64_t i = 0; i < 4 && offset < Size; i++) {
                for (int64_t j = 0; j < 4 && offset < Size; j++) {
                    base[i][j] = static_cast<float>(Data[offset++]);
                }
            }
            
            // Create non-contiguous tensor via transpose
            torch::Tensor transposed = base.t();
            torch::Tensor result_transposed = torch::msort(transposed);
            
            // Slice creates non-contiguous tensor
            torch::Tensor sliced = base.slice(0, 0, 2);
            torch::Tensor result_sliced = torch::msort(sliced);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}