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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a conjugated version of the tensor
        torch::Tensor conj_tensor = input_tensor.conj();
        
        // Apply resolve_conj to the original tensor
        torch::Tensor resolved_original = torch::resolve_conj(input_tensor);
        
        // Apply resolve_conj to the conjugated tensor
        torch::Tensor resolved_conj = torch::resolve_conj(conj_tensor);
        
        // Try resolve_conj on a view of the tensor
        if (input_tensor.dim() > 0 && input_tensor.numel() > 0) {
            torch::Tensor view_tensor = input_tensor.view({-1});
            torch::Tensor resolved_view = torch::resolve_conj(view_tensor);
        }
        
        // Try resolve_conj on a sliced tensor if possible
        if (input_tensor.dim() > 0 && input_tensor.size(0) > 1) {
            torch::Tensor sliced = input_tensor.slice(0, 0, input_tensor.size(0) - 1);
            torch::Tensor resolved_slice = torch::resolve_conj(sliced);
        }
        
        // Try resolve_conj on a transposed tensor if possible
        if (input_tensor.dim() >= 2) {
            torch::Tensor transposed = input_tensor.transpose(0, 1);
            torch::Tensor resolved_transpose = torch::resolve_conj(transposed);
        }
        
        // Try resolve_conj on a tensor with requires_grad set
        if (input_tensor.dtype() == torch::kFloat || 
            input_tensor.dtype() == torch::kDouble || 
            input_tensor.dtype() == torch::kComplexFloat || 
            input_tensor.dtype() == torch::kComplexDouble) {
            
            auto grad_tensor = input_tensor.clone().detach().requires_grad_(true);
            torch::Tensor resolved_grad = torch::resolve_conj(grad_tensor);
        }
        
        // Try resolve_conj on a zero-sized tensor
        if (offset + 2 < Size) {
            std::vector<int64_t> empty_shape = {0};
            torch::Tensor empty_tensor = torch::empty(empty_shape, torch::TensorOptions().dtype(fuzzer_utils::parseDataType(Data[offset])));
            torch::Tensor resolved_empty = torch::resolve_conj(empty_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
