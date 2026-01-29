#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstring>        // For memcpy

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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse diagonal parameter if we have more data
        int64_t diagonal = 0;
        if (offset + sizeof(int8_t) <= Size) {
            // Use int8_t to get reasonable diagonal values (-128 to 127)
            int8_t diag_byte;
            std::memcpy(&diag_byte, Data + offset, sizeof(int8_t));
            diagonal = static_cast<int64_t>(diag_byte);
            offset += sizeof(int8_t);
        }
        
        // Apply torch.tril operation
        // torch::tril returns the lower triangular part of a matrix (2-D tensor)
        // or batch of matrices. For tensors with more than 2 dimensions,
        // it operates on the last two dimensions.
        torch::Tensor result = torch::tril(input_tensor, diagonal);
        
        // Try another variant with different diagonal value if we have more data
        if (offset + sizeof(int8_t) <= Size) {
            int8_t diag_byte2;
            std::memcpy(&diag_byte2, Data + offset, sizeof(int8_t));
            int64_t diagonal2 = static_cast<int64_t>(diag_byte2);
            offset += sizeof(int8_t);
            torch::Tensor result2 = torch::tril(input_tensor, diagonal2);
        }
        
        // Try in-place variant
        try {
            torch::Tensor input_copy = input_tensor.clone();
            input_copy.tril_(diagonal);
        } catch (const std::exception&) {
            // Ignore exceptions from in-place operation (e.g., dimension issues)
        }
        
        // Try with 2D tensor specifically (tril is designed for 2D or batched 2D)
        try {
            if (input_tensor.numel() >= 4) {
                int64_t side = static_cast<int64_t>(std::sqrt(input_tensor.numel()));
                if (side >= 2) {
                    torch::Tensor tensor_2d = input_tensor.flatten().narrow(0, 0, side * side).view({side, side});
                    torch::Tensor result_2d = torch::tril(tensor_2d, diagonal);
                }
            }
        } catch (const std::exception&) {
            // Ignore reshape exceptions
        }
        
        // Try with 3D tensor (batch of matrices)
        try {
            if (input_tensor.numel() >= 8) {
                torch::Tensor tensor_3d = input_tensor.flatten().narrow(0, 0, 8).view({2, 2, 2});
                torch::Tensor result_3d = torch::tril(tensor_3d, diagonal);
            }
        } catch (const std::exception&) {
            // Ignore reshape exceptions
        }
        
        // Try with empty 2D tensor
        try {
            torch::Tensor empty_tensor = torch::empty({0, 0}, input_tensor.options());
            torch::Tensor empty_result = torch::tril(empty_tensor, diagonal);
        } catch (const std::exception&) {
            // Ignore exceptions from empty tensor
        }
        
        // Try with different dtypes
        try {
            torch::Tensor float_tensor = input_tensor.to(torch::kFloat32);
            torch::Tensor float_result = torch::tril(float_tensor, diagonal);
        } catch (const std::exception&) {
            // Ignore dtype conversion exceptions
        }
        
        try {
            torch::Tensor int_tensor = input_tensor.to(torch::kInt32);
            torch::Tensor int_result = torch::tril(int_tensor, diagonal);
        } catch (const std::exception&) {
            // Ignore dtype conversion exceptions
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}