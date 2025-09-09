#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic data for tensor creation
        if (Size < 32) {
            return 0;
        }

        // Extract tensor shapes and properties
        auto input_shape = extract_tensor_shape(Data, Size, offset, 4);
        auto tensor1_shape = extract_tensor_shape(Data, Size, offset, 4);
        auto tensor2_shape = extract_tensor_shape(Data, Size, offset, 4);
        
        if (offset >= Size) return 0;

        // Extract dtype
        auto dtype = extract_dtype(Data, Size, offset);
        
        // Extract value scalar
        double value = 1.0;
        if (offset + sizeof(float) <= Size) {
            value = extract_float_from_bytes(Data + offset);
            offset += sizeof(float);
        }

        // Create input tensors with different initialization strategies
        torch::Tensor input, tensor1, tensor2;
        
        // Test different tensor creation methods
        uint8_t creation_method = (offset < Size) ? Data[offset++] % 4 : 0;
        
        switch (creation_method) {
            case 0:
                // Random tensors
                input = torch::randn(input_shape, torch::dtype(dtype));
                tensor1 = torch::randn(tensor1_shape, torch::dtype(dtype));
                tensor2 = torch::randn(tensor2_shape, torch::dtype(dtype));
                break;
            case 1:
                // Ones tensors
                input = torch::ones(input_shape, torch::dtype(dtype));
                tensor1 = torch::ones(tensor1_shape, torch::dtype(dtype));
                tensor2 = torch::ones(tensor2_shape, torch::dtype(dtype));
                break;
            case 2:
                // Zeros tensors
                input = torch::zeros(input_shape, torch::dtype(dtype));
                tensor1 = torch::zeros(tensor1_shape, torch::dtype(dtype));
                tensor2 = torch::zeros(tensor2_shape, torch::dtype(dtype));
                break;
            case 3:
                // Mixed: some from data, some random
                input = create_tensor_from_data(Data, Size, offset, input_shape, dtype);
                tensor1 = torch::randn(tensor1_shape, torch::dtype(dtype));
                tensor2 = torch::randn(tensor2_shape, torch::dtype(dtype));
                break;
        }

        // Test edge cases for value parameter
        if (offset < Size) {
            uint8_t value_case = Data[offset++] % 6;
            switch (value_case) {
                case 0: value = 0.0; break;
                case 1: value = 1.0; break;
                case 2: value = -1.0; break;
                case 3: value = std::numeric_limits<double>::infinity(); break;
                case 4: value = -std::numeric_limits<double>::infinity(); break;
                case 5: value = std::numeric_limits<double>::quiet_NaN(); break;
            }
        }

        // Test different function call variants
        uint8_t call_variant = (offset < Size) ? Data[offset++] % 4 : 0;
        
        torch::Tensor result;
        
        switch (call_variant) {
            case 0:
                // Basic call with default value
                result = torch::addcmul(input, tensor1, tensor2);
                break;
            case 1:
                // Call with explicit value
                result = torch::addcmul(input, tensor1, tensor2, value);
                break;
            case 2:
                // Call with output tensor
                {
                    torch::Tensor out = torch::empty_like(input);
                    result = torch::addcmul_out(out, input, tensor1, tensor2, value);
                }
                break;
            case 3:
                // In-place operation
                {
                    torch::Tensor input_copy = input.clone();
                    result = input_copy.addcmul_(tensor1, tensor2, value);
                }
                break;
        }

        // Test broadcasting edge cases
        if (offset < Size && Data[offset++] % 2 == 0) {
            // Test with scalar tensors
            auto scalar1 = torch::tensor(1.5, torch::dtype(dtype));
            auto scalar2 = torch::tensor(2.0, torch::dtype(dtype));
            torch::addcmul(input, scalar1, scalar2, value);
        }

        // Test with different device placements if CUDA is available
        if (torch::cuda::is_available() && offset < Size && Data[offset++] % 2 == 0) {
            auto cuda_input = input.to(torch::kCUDA);
            auto cuda_tensor1 = tensor1.to(torch::kCUDA);
            auto cuda_tensor2 = tensor2.to(torch::kCUDA);
            torch::addcmul(cuda_input, cuda_tensor1, cuda_tensor2, value);
        }

        // Test gradient computation if tensors require grad
        if (dtype == torch::kFloat || dtype == torch::kDouble) {
            if (offset < Size && Data[offset++] % 2 == 0) {
                input.requires_grad_(true);
                tensor1.requires_grad_(true);
                tensor2.requires_grad_(true);
                
                auto grad_result = torch::addcmul(input, tensor1, tensor2, value);
                auto loss = grad_result.sum();
                loss.backward();
            }
        }

        // Test with extreme tensor shapes for broadcasting
        if (offset < Size && Data[offset++] % 3 == 0) {
            // Test with 1D tensors
            auto vec1 = torch::randn({10}, torch::dtype(dtype));
            auto vec2 = torch::randn({1}, torch::dtype(dtype));
            auto vec_input = torch::randn({10}, torch::dtype(dtype));
            torch::addcmul(vec_input, vec1, vec2, value);
        }

        // Test with empty tensors
        if (offset < Size && Data[offset++] % 4 == 0) {
            auto empty_input = torch::empty({0}, torch::dtype(dtype));
            auto empty_t1 = torch::empty({0}, torch::dtype(dtype));
            auto empty_t2 = torch::empty({0}, torch::dtype(dtype));
            torch::addcmul(empty_input, empty_t1, empty_t2, value);
        }

        // Force evaluation of result to catch any lazy evaluation issues
        if (result.defined()) {
            result.sum().item<double>();
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}