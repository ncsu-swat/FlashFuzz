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
        
        if (Size < 4) {
            return 0;
        }
        
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            tensor2 = torch::ones_like(tensor1);
        }
        
        // Avoid division by zero in tensor2
        tensor2 = torch::where(tensor2 == 0, torch::ones_like(tensor2), tensor2);
        
        torch::Tensor result;
        
        uint8_t variant = 0;
        if (offset < Size) {
            variant = Data[offset++] % 4;
        }
        
        if (variant == 0) {
            // Variant 1: floor_divide with scalar
            double scalar_value = 1.0;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&scalar_value, Data + offset, sizeof(double));
                offset += sizeof(double);
            }
            
            // Avoid division by zero and NaN/Inf
            if (scalar_value == 0.0 || std::isnan(scalar_value) || std::isinf(scalar_value)) {
                scalar_value = 1.0;
            }
            
            // Use div with floor rounding mode (modern equivalent)
            result = torch::div(tensor1, torch::scalar_tensor(scalar_value), "floor");
        } else if (variant == 1) {
            // Variant 2: floor_divide with tensor using the function
            result = torch::floor_divide(tensor1, tensor2);
        } else if (variant == 2) {
            // Variant 3: Use div with floor rounding mode (recommended modern API)
            result = torch::div(tensor1, tensor2, "floor");
        } else {
            // Variant 4: Integer division path
            torch::Tensor int_tensor1 = tensor1.to(torch::kInt32);
            torch::Tensor int_tensor2 = tensor2.to(torch::kInt32);
            int_tensor2 = torch::where(int_tensor2 == 0, torch::ones_like(int_tensor2), int_tensor2);
            result = torch::floor_divide(int_tensor1, int_tensor2);
        }
        
        // Verify result is valid
        auto sizes = result.sizes();
        auto dtype = result.dtype();
        auto numel = result.numel();
        
        if (numel > 0 && numel == 1) {
            (void)result.item<float>();
        }
        
        // Test broadcasting with different shapes
        if (offset < Size) {
            torch::Tensor tensor3 = fuzzer_utils::createTensor(Data, Size, offset);
            tensor3 = torch::where(tensor3 == 0, torch::ones_like(tensor3), tensor3);
            
            try {
                torch::Tensor broadcast_result = torch::floor_divide(tensor1, tensor3);
                (void)broadcast_result.sizes();
            } catch (const c10::Error &) {
                // Broadcasting might fail due to incompatible shapes
            }
        }
        
        // Test with output tensor
        if (offset < Size && Data[offset] % 2 == 0) {
            try {
                torch::Tensor out_tensor = torch::empty_like(tensor1);
                torch::floor_divide_out(out_tensor, tensor1, tensor2);
                (void)out_tensor.numel();
            } catch (const c10::Error &) {
                // May fail with incompatible shapes
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