#include "fuzzer_utils.h"
#include <iostream>
#include <limits>

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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create first input tensor
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor if we have more data
        torch::Tensor input2;
        if (offset < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no more data, create a tensor with same shape but different values
            input2 = torch::randn_like(input1.to(torch::kFloat));
        }
        
        // logaddexp requires floating point tensors
        if (!input1.is_floating_point()) {
            input1 = input1.to(torch::kFloat);
        }
        if (!input2.is_floating_point()) {
            input2 = input2.to(torch::kFloat);
        }
        
        // Ensure both have the same floating point type
        if (input1.scalar_type() != input2.scalar_type()) {
            if (input1.scalar_type() == torch::kDouble || input2.scalar_type() == torch::kDouble) {
                input1 = input1.to(torch::kDouble);
                input2 = input2.to(torch::kDouble);
            } else {
                input1 = input1.to(torch::kFloat);
                input2 = input2.to(torch::kFloat);
            }
        }
        
        // Handle shape compatibility for broadcasting
        try {
            // Check if broadcasting works by doing a dummy add
            auto dummy = input1 + input2;
            (void)dummy;
        } catch (...) {
            // If broadcasting fails, reshape or recreate second tensor
            if (input1.numel() == input2.numel()) {
                input2 = input2.reshape(input1.sizes());
            } else {
                input2 = torch::randn_like(input1);
            }
        }
        
        // Apply logaddexp operation
        torch::Tensor result = torch::logaddexp(input1, input2);
        
        // Ensure result is computed
        if (result.defined()) {
            (void)result.sum().item<float>();
        }
        
        // Test with extreme values based on fuzzer data
        if (offset < Size) {
            uint8_t selector = Data[offset++];
            
            auto extreme_tensor = torch::zeros_like(input1);
            
            switch (selector % 5) {
                case 0:
                    // Very large positive values
                    extreme_tensor.fill_(1e38f);
                    break;
                case 1:
                    // Very large negative values
                    extreme_tensor.fill_(-1e38f);
                    break;
                case 2:
                    // Positive infinity
                    extreme_tensor.fill_(std::numeric_limits<float>::infinity());
                    break;
                case 3:
                    // Negative infinity
                    extreme_tensor.fill_(-std::numeric_limits<float>::infinity());
                    break;
                case 4:
                    // NaN
                    extreme_tensor.fill_(std::numeric_limits<float>::quiet_NaN());
                    break;
            }
            
            // logaddexp with extreme values (expected to work without throwing)
            torch::Tensor extreme_result = torch::logaddexp(extreme_tensor, input2);
            (void)extreme_result.numel();
        }
        
        // Test with zero-sized dimensions
        if (offset < Size && (Data[offset++] % 3 == 0)) {
            auto zero_tensor1 = torch::empty({0, 2}, input1.options());
            auto zero_tensor2 = torch::empty({0, 2}, input2.options());
            
            torch::Tensor zero_result = torch::logaddexp(zero_tensor1, zero_tensor2);
            (void)zero_result.numel();
        }
        
        // Test with scalar tensors
        if (offset < Size && (Data[offset++] % 2 == 0)) {
            auto scalar1 = torch::tensor(1.5f);
            auto scalar2 = torch::tensor(2.5f);
            
            torch::Tensor scalar_result = torch::logaddexp(scalar1, scalar2);
            (void)scalar_result.item<float>();
        }
        
        // Test logaddexp with out parameter
        if (offset < Size && (Data[offset++] % 2 == 0)) {
            torch::Tensor out_tensor = torch::empty_like(result);
            torch::logaddexp_out(out_tensor, input1, input2);
            (void)out_tensor.numel();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}