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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create first tensor (batch1 x n x m)
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 3 dimensions for bmm
        if (input1.dim() < 3) {
            while (input1.dim() < 3) {
                input1 = input1.unsqueeze(0);
            }
        }
        
        // Create second tensor (batch2 x m x p)
        torch::Tensor input2;
        if (offset < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure we have at least 3 dimensions for bmm
            if (input2.dim() < 3) {
                while (input2.dim() < 3) {
                    input2 = input2.unsqueeze(0);
                }
            }
            
            // Ensure compatible dimensions for matrix multiplication
            // For bmm: input1.shape = [batch, n, m], input2.shape = [batch, m, p]
            int64_t batch1 = input1.size(0);
            int64_t m1 = input1.size(2);
            
            int64_t batch2 = input2.size(0);
            int64_t m2 = input2.size(1);
            int64_t p = input2.size(2);
            
            // Reshape input2 to match batch size and middle dimension if needed
            if (batch1 != batch2 || m1 != m2) {
                auto options = torch::TensorOptions().dtype(input1.dtype());
                input2 = torch::ones({batch1, m1, p}, options);
            } else {
                // Ensure same dtype
                input2 = input2.to(input1.dtype());
            }
        } else {
            // If we don't have enough data for the second tensor, create a compatible one
            int64_t batch = input1.size(0);
            int64_t m = input1.size(2);
            int64_t p = 1;
            
            auto options = torch::TensorOptions().dtype(input1.dtype());
            input2 = torch::ones({batch, m, p}, options);
        }
        
        // Ensure both tensors are floating point for bmm
        if (!input1.is_floating_point()) {
            input1 = input1.to(torch::kFloat);
            input2 = input2.to(torch::kFloat);
        }
        
        // Apply bmm operation
        torch::Tensor output = torch::bmm(input1, input2);
        
        // Test edge cases with zero dimensions
        if (offset + 1 < Size) {
            uint8_t edge_case = Data[offset++];
            
            try {
                if (edge_case % 4 == 0) {
                    // Test with zero batch dimension
                    auto options = torch::TensorOptions().dtype(torch::kFloat);
                    torch::Tensor zero_batch1 = torch::ones({0, 2, 3}, options);
                    torch::Tensor zero_batch2 = torch::ones({0, 3, 2}, options);
                    torch::Tensor zero_output = torch::bmm(zero_batch1, zero_batch2);
                } else if (edge_case % 4 == 1) {
                    // Test with zero inner dimensions
                    auto options = torch::TensorOptions().dtype(torch::kFloat);
                    torch::Tensor zero_inner1 = torch::ones({2, 3, 0}, options);
                    torch::Tensor zero_inner2 = torch::ones({2, 0, 3}, options);
                    torch::Tensor zero_output = torch::bmm(zero_inner1, zero_inner2);
                } else if (edge_case % 4 == 2) {
                    // Test with double precision
                    auto options = torch::TensorOptions().dtype(torch::kDouble);
                    torch::Tensor double_tensor1 = torch::randn({2, 3, 4}, options);
                    torch::Tensor double_tensor2 = torch::randn({2, 4, 3}, options);
                    torch::Tensor double_output = torch::bmm(double_tensor1, double_tensor2);
                } else {
                    // Test with moderate dimensions derived from fuzz data
                    if (offset + 2 <= Size) {
                        int64_t dim1 = (Data[offset] % 50) + 1;
                        int64_t dim2 = (Data[offset + 1] % 50) + 1;
                        offset += 2;
                        
                        auto options = torch::TensorOptions().dtype(torch::kFloat);
                        torch::Tensor varied1 = torch::randn({2, dim1, dim2}, options);
                        torch::Tensor varied2 = torch::randn({2, dim2, dim1}, options);
                        torch::Tensor varied_output = torch::bmm(varied1, varied2);
                    }
                }
            } catch (const std::exception&) {
                // Expected failures for edge cases - catch silently
            }
        }
        
        // Test out parameter variant if available
        if (offset < Size && (Data[offset] % 2 == 0)) {
            try {
                auto options = torch::TensorOptions().dtype(input1.dtype());
                torch::Tensor out_tensor = torch::empty({input1.size(0), input1.size(1), input2.size(2)}, options);
                torch::bmm_out(out_tensor, input1, input2);
            } catch (const std::exception&) {
                // Silently catch expected failures
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