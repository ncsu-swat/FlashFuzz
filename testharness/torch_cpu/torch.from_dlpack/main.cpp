#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert tensor to DLPack format
        auto dlpack = input_tensor.to_dlpack();
        
        // Convert back from DLPack to PyTorch tensor
        torch::Tensor output_tensor = torch::from_dlpack(dlpack);
        
        // Verify the conversion worked correctly
        if (input_tensor.sizes() != output_tensor.sizes() ||
            input_tensor.dtype() != output_tensor.dtype()) {
            throw std::runtime_error("DLPack conversion changed tensor properties");
        }
        
        // Test with different tensor properties
        if (offset + 1 < Size) {
            // Create another tensor with different properties
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try to convert it to DLPack and back
            auto another_dlpack = another_tensor.to_dlpack();
            torch::Tensor another_output = torch::from_dlpack(another_dlpack);
            
            // Try operations on the converted tensor
            if (another_output.numel() > 0) {
                auto sum = another_output.sum();
                auto mean = another_output.mean();
            }
        }
        
        // Test with empty tensor if we have enough data
        if (offset + 1 < Size) {
            torch::Tensor empty_tensor = torch::empty({0});
            auto empty_dlpack = empty_tensor.to_dlpack();
            torch::Tensor empty_output = torch::from_dlpack(empty_dlpack);
        }
        
        // Test with scalar tensor
        if (offset + 1 < Size) {
            torch::Tensor scalar_tensor = torch::tensor(3.14);
            auto scalar_dlpack = scalar_tensor.to_dlpack();
            torch::Tensor scalar_output = torch::from_dlpack(scalar_dlpack);
        }
        
        // Test with boolean tensor
        if (offset + 1 < Size) {
            torch::Tensor bool_tensor = torch::tensor(true);
            auto bool_dlpack = bool_tensor.to_dlpack();
            torch::Tensor bool_output = torch::from_dlpack(bool_dlpack);
        }
        
        // Test with complex tensor if we have enough data
        if (offset + 2 < Size) {
            torch::Tensor complex_tensor = torch::complex(
                torch::ones({2, 2}), 
                torch::ones({2, 2})
            );
            auto complex_dlpack = complex_tensor.to_dlpack();
            torch::Tensor complex_output = torch::from_dlpack(complex_dlpack);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}