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
        
        // Apply flipud operation
        torch::Tensor result = torch::flipud(input_tensor);
        
        // Try to access elements to ensure computation is performed
        if (result.numel() > 0) {
            result.item();
        }
        
        // Try with a view of the tensor to test different memory layouts
        if (input_tensor.dim() > 0 && input_tensor.numel() > 0) {
            torch::Tensor view_tensor = input_tensor.transpose(0, input_tensor.dim() - 1);
            torch::Tensor view_result = torch::flipud(view_tensor);
            
            if (view_result.numel() > 0) {
                view_result.item();
            }
        }
        
        // Try with non-contiguous tensor
        if (input_tensor.dim() > 1 && input_tensor.size(0) > 1 && input_tensor.numel() > 0) {
            torch::Tensor non_contig = input_tensor.transpose(0, input_tensor.dim() - 1);
            torch::Tensor non_contig_result = torch::flipud(non_contig);
            
            if (non_contig_result.numel() > 0) {
                non_contig_result.item();
            }
        }
        
        // Try with empty tensor
        if (offset + 2 < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            torch::Tensor empty_tensor = torch::empty({0}, torch::TensorOptions().dtype(dtype));
            torch::Tensor empty_result = torch::flipud(empty_tensor);
        }
        
        // Try with scalar tensor
        if (offset + 2 < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            torch::Tensor scalar_tensor = torch::tensor(42, torch::TensorOptions().dtype(dtype));
            torch::Tensor scalar_result = torch::flipud(scalar_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
