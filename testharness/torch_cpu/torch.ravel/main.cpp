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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.ravel operation
        // torch.ravel returns a contiguous flattened tensor
        torch::Tensor result = input_tensor.ravel();
        
        // Verify the result has the expected properties
        // The result should be 1D and have the same number of elements as the input
        if (result.dim() != 1 || result.numel() != input_tensor.numel()) {
            throw std::runtime_error("Ravel operation produced unexpected result");
        }
        
        // Try to access elements to ensure the tensor is valid
        if (result.numel() > 0) {
            result[0].item();
        }
        
        // Try alternative ways to call ravel
        torch::Tensor result2 = torch::ravel(input_tensor);
        
        // Try with a view of the tensor
        if (input_tensor.numel() > 0) {
            torch::Tensor view_tensor = input_tensor.view({-1});
            torch::Tensor result3 = view_tensor.ravel();
        }
        
        // Try with a non-contiguous tensor if possible
        if (input_tensor.dim() > 1 && input_tensor.size(0) > 1 && input_tensor.size(1) > 1) {
            torch::Tensor transposed = input_tensor.transpose(0, 1);
            torch::Tensor result4 = transposed.ravel();
        }
        
        // Try with a zero-sized tensor
        if (offset + 2 < Size) {
            std::vector<int64_t> empty_shape = {0};
            torch::Tensor empty_tensor = torch::empty(empty_shape, input_tensor.options());
            torch::Tensor result5 = empty_tensor.ravel();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
