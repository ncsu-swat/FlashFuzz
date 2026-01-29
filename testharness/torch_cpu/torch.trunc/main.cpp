#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // torch::trunc only works on floating-point tensors
        if (!input_tensor.is_floating_point()) {
            input_tensor = input_tensor.to(torch::kFloat32);
        }
        
        // Apply torch.trunc operation
        torch::Tensor result = torch::trunc(input_tensor);
        
        // Try in-place version if there's enough data to decide
        if (offset < Size) {
            bool use_inplace = Data[offset++] % 2 == 0;
            if (use_inplace) {
                torch::Tensor input_copy = input_tensor.clone();
                input_copy.trunc_();
            }
        }
        
        // Try with different output tensor if there's enough data
        if (offset < Size) {
            bool use_out = Data[offset++] % 2 == 0;
            if (use_out) {
                torch::Tensor output = torch::empty_like(input_tensor);
                torch::trunc_out(output, input_tensor);
            }
        }
        
        // Test with different floating point dtypes
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 3;
            torch::Tensor typed_input;
            try {
                switch (dtype_selector) {
                    case 0:
                        typed_input = input_tensor.to(torch::kFloat32);
                        break;
                    case 1:
                        typed_input = input_tensor.to(torch::kFloat64);
                        break;
                    case 2:
                        typed_input = input_tensor.to(torch::kFloat16);
                        break;
                }
                torch::Tensor typed_result = torch::trunc(typed_input);
            } catch (...) {
                // Some dtypes may not be supported on all platforms
            }
        }
        
        // Test with special floating point values
        if (offset < Size && input_tensor.numel() > 0) {
            bool test_special = Data[offset++] % 2 == 0;
            if (test_special) {
                torch::Tensor special_tensor = torch::zeros_like(input_tensor);
                auto accessor = special_tensor.accessor<float, 1>();
                if (special_tensor.dim() == 1 && special_tensor.size(0) >= 3) {
                    // This may not work for all tensor shapes, wrap in try-catch
                    try {
                        special_tensor[0] = std::numeric_limits<float>::infinity();
                        special_tensor[1] = -std::numeric_limits<float>::infinity();
                        special_tensor[2] = std::nanf("");
                        torch::Tensor special_result = torch::trunc(special_tensor);
                    } catch (...) {
                        // Shape mismatch or other issues
                    }
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}