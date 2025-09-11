#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;

        // Need at least 3 tensors for addcdiv: input, tensor1, tensor2
        if (Size < 6) // Minimum bytes needed for basic tensor creation
            return 0;

        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);

        // Create tensor1 for division
        if (offset >= Size)
            return 0;
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);

        // Create tensor2 for division (denominator)
        if (offset >= Size)
            return 0;
        torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);

        // Parse value for scaling factor
        float value = 1.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&value, Data + offset, sizeof(float));
            offset += sizeof(float);
        }

        // Try different variants of addcdiv
        try {
            // Basic addcdiv: input + value * (tensor1 / tensor2)
            torch::Tensor result1 = torch::addcdiv(input, tensor1, tensor2, value);
        } catch (const std::exception&) {
            // Catch and continue - we want to test other variants
        }

        try {
            // Variant with default value (1.0)
            torch::Tensor result2 = torch::addcdiv(input, tensor1, tensor2);
        } catch (const std::exception&) {
            // Catch and continue
        }

        try {
            // In-place variant
            torch::Tensor input_copy = input.clone();
            input_copy.addcdiv_(tensor1, tensor2, value);
        } catch (const std::exception&) {
            // Catch and continue
        }

        try {
            // In-place with default value
            torch::Tensor input_copy = input.clone();
            input_copy.addcdiv_(tensor1, tensor2);
        } catch (const std::exception&) {
            // Catch and continue
        }

        try {
            // Out variant
            torch::Tensor output = torch::empty_like(input);
            torch::addcdiv_out(output, input, tensor1, tensor2, value);
        } catch (const std::exception&) {
            // Catch and continue
        }

        // Test with scalar tensors
        try {
            if (input.numel() > 0 && tensor1.numel() > 0 && tensor2.numel() > 0) {
                torch::Tensor scalar_input = input.flatten()[0].unsqueeze(0);
                torch::Tensor scalar_t1 = tensor1.flatten()[0].unsqueeze(0);
                torch::Tensor scalar_t2 = tensor2.flatten()[0].unsqueeze(0);
                
                torch::Tensor scalar_result = torch::addcdiv(scalar_input, scalar_t1, scalar_t2, value);
            }
        } catch (const std::exception&) {
            // Catch and continue
        }

        // Test with empty tensors
        try {
            torch::Tensor empty_tensor = torch::empty({0}, input.options());
            if (input.dim() > 0) {
                std::vector<int64_t> empty_shape = input.sizes().vec();
                empty_shape[0] = 0;
                torch::Tensor shaped_empty = torch::empty(empty_shape, input.options());
                torch::Tensor result_empty = torch::addcdiv(shaped_empty, tensor1, tensor2, value);
            }
        } catch (const std::exception&) {
            // Catch and continue
        }

        // Test with extreme values
        try {
            torch::Tensor extreme_input = torch::full_like(input, std::numeric_limits<float>::max());
            torch::Tensor extreme_t1 = torch::full_like(tensor1, std::numeric_limits<float>::max());
            torch::Tensor extreme_t2 = torch::full_like(tensor2, std::numeric_limits<float>::min());
            
            torch::Tensor extreme_result = torch::addcdiv(extreme_input, extreme_t1, extreme_t2, value);
        } catch (const std::exception&) {
            // Catch and continue
        }

        // Test with zero denominator
        try {
            torch::Tensor zero_tensor = torch::zeros_like(tensor2);
            torch::Tensor div_by_zero = torch::addcdiv(input, tensor1, zero_tensor, value);
        } catch (const std::exception&) {
            // Expected to throw
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
