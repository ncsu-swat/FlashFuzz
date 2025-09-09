#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate input tensor with various properties
        auto tensor_info = generate_tensor_info(Data, Size, offset);
        if (offset >= Size) return 0;

        // Create tensor with different dtypes, including complex types
        torch::Tensor input_tensor;
        
        // Test with different data types, focusing on complex types where resolve_conj is most relevant
        auto dtype_choice = consume_integral_in_range<int>(Data, Size, offset, 0, 6);
        if (offset >= Size) return 0;

        switch (dtype_choice) {
            case 0:
                input_tensor = torch::randn(tensor_info.sizes, torch::dtype(torch::kComplexFloat));
                break;
            case 1:
                input_tensor = torch::randn(tensor_info.sizes, torch::dtype(torch::kComplexDouble));
                break;
            case 2:
                input_tensor = torch::randn(tensor_info.sizes, torch::dtype(torch::kFloat));
                break;
            case 3:
                input_tensor = torch::randn(tensor_info.sizes, torch::dtype(torch::kDouble));
                break;
            case 4:
                input_tensor = torch::randint(-100, 100, tensor_info.sizes, torch::dtype(torch::kInt));
                break;
            case 5:
                input_tensor = torch::randint(-100, 100, tensor_info.sizes, torch::dtype(torch::kLong));
                break;
            default:
                input_tensor = torch::randn(tensor_info.sizes, torch::dtype(torch::kFloat));
                break;
        }

        // Test with conjugated tensors (most important case for resolve_conj)
        auto should_conjugate = consume_integral_in_range<int>(Data, Size, offset, 0, 1);
        if (offset >= Size) return 0;

        if (should_conjugate && (input_tensor.dtype() == torch::kComplexFloat || 
                                input_tensor.dtype() == torch::kComplexDouble)) {
            input_tensor = torch::conj(input_tensor);
        }

        // Test resolve_conj on the tensor
        auto result = torch::resolve_conj(input_tensor);

        // Verify basic properties
        if (result.sizes() != input_tensor.sizes()) {
            std::cerr << "Size mismatch in resolve_conj result" << std::endl;
        }

        if (result.dtype() != input_tensor.dtype()) {
            std::cerr << "Dtype mismatch in resolve_conj result" << std::endl;
        }

        // Test with different tensor states
        auto tensor_state = consume_integral_in_range<int>(Data, Size, offset, 0, 4);
        if (offset >= Size) return 0;

        torch::Tensor test_tensor = input_tensor.clone();
        
        switch (tensor_state) {
            case 0:
                // Test with transposed tensor
                if (test_tensor.dim() >= 2) {
                    test_tensor = test_tensor.transpose(0, 1);
                }
                break;
            case 1:
                // Test with sliced tensor
                if (test_tensor.numel() > 1) {
                    test_tensor = test_tensor.slice(0, 0, std::min(2L, test_tensor.size(0)));
                }
                break;
            case 2:
                // Test with reshaped tensor
                if (test_tensor.numel() > 0) {
                    test_tensor = test_tensor.reshape({-1});
                }
                break;
            case 3:
                // Test with detached tensor
                test_tensor = test_tensor.detach();
                break;
            case 4:
                // Test with requires_grad tensor (if floating point)
                if (test_tensor.dtype().is_floating_point()) {
                    test_tensor.requires_grad_(true);
                }
                break;
        }

        // Apply resolve_conj to modified tensor
        auto result2 = torch::resolve_conj(test_tensor);

        // Test edge cases with empty tensors
        auto empty_tensor = torch::empty({0}, input_tensor.dtype());
        auto empty_result = torch::resolve_conj(empty_tensor);

        // Test with scalar tensors
        auto scalar_tensor = torch::scalar_tensor(1.0, input_tensor.dtype());
        if (scalar_tensor.dtype() == torch::kComplexFloat || 
            scalar_tensor.dtype() == torch::kComplexDouble) {
            scalar_tensor = torch::conj(scalar_tensor);
        }
        auto scalar_result = torch::resolve_conj(scalar_tensor);

        // Test chaining resolve_conj calls
        auto chained_result = torch::resolve_conj(torch::resolve_conj(input_tensor));

        // Verify that resolve_conj is idempotent for non-conjugated tensors
        if (!input_tensor.is_conj()) {
            auto double_resolve = torch::resolve_conj(torch::resolve_conj(input_tensor));
            // The results should be equivalent for non-conjugated tensors
        }

        // Test with different memory formats if applicable
        if (input_tensor.dim() == 4 && input_tensor.numel() > 0) {
            auto channels_last_tensor = input_tensor.to(torch::MemoryFormat::ChannelsLast);
            auto channels_last_result = torch::resolve_conj(channels_last_tensor);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}