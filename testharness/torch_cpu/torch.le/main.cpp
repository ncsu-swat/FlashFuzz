#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

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
        
        if (Size < 4) {
            return 0;
        }
        
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            uint8_t scalar_value = Data[Size - 1];
            tensor2 = torch::tensor(static_cast<float>(scalar_value), tensor1.options());
        }
        
        // 1. Element-wise comparison (tensor1 <= tensor2)
        try {
            torch::Tensor result1 = torch::le(tensor1, tensor2);
        } catch (const std::exception& e) {
            // Shape mismatch is expected, continue
        }
        
        // 2. Tensor-scalar comparison using a value from fuzzer data
        try {
            float scalar_value = 0.0f;
            if (offset < Size) {
                scalar_value = static_cast<float>(Data[offset % Size]);
            }
            torch::Tensor result2 = torch::le(tensor1, torch::Scalar(scalar_value));
        } catch (const std::exception& e) {
            // Continue
        }
        
        // 3. In-place version (tensor1.le_(tensor2))
        try {
            torch::Tensor tensor1_copy = tensor1.clone();
            tensor1_copy.le_(tensor2);
        } catch (const std::exception& e) {
            // Continue
        }
        
        // 4. In-place version with scalar
        try {
            torch::Tensor tensor1_copy = tensor1.clone();
            float scalar_value = static_cast<float>(Data[0]);
            tensor1_copy.le_(torch::Scalar(scalar_value));
        } catch (const std::exception& e) {
            // Continue
        }
        
        // 5. Out version with tensor
        try {
            auto result_shape = tensor1.sizes().vec();
            if (tensor2.dim() > 0) {
                // Try to compute broadcast shape
                for (size_t i = 0; i < std::min(result_shape.size(), static_cast<size_t>(tensor2.dim())); i++) {
                    size_t idx1 = result_shape.size() - 1 - i;
                    size_t idx2 = tensor2.dim() - 1 - i;
                    if (result_shape[idx1] < tensor2.sizes()[idx2]) {
                        result_shape[idx1] = tensor2.sizes()[idx2];
                    }
                }
            }
            torch::Tensor out = torch::empty(result_shape, tensor1.options().dtype(torch::kBool));
            torch::le_out(out, tensor1, tensor2);
        } catch (const std::exception& e) {
            // Continue
        }
        
        // 6. Test with broadcasting - single element tensor
        try {
            torch::Tensor broadcast_tensor = torch::tensor({1.0f});
            torch::Tensor result4 = torch::le(tensor1, broadcast_tensor);
        } catch (const std::exception& e) {
            // Continue
        }
        
        // 7. Test with different dtypes
        try {
            if (tensor1.is_floating_point()) {
                torch::Tensor tensor2_double = tensor2.to(torch::kDouble);
                torch::Tensor result5 = torch::le(tensor1.to(torch::kDouble), tensor2_double);
            }
        } catch (const std::exception& e) {
            // Continue
        }
        
        // 8. Test comparison with zero tensor
        try {
            torch::Tensor zeros = torch::zeros_like(tensor1);
            torch::Tensor result6 = torch::le(tensor1, zeros);
        } catch (const std::exception& e) {
            // Continue
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}