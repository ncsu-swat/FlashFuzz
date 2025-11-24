#include "fuzzer_utils.h" // General fuzzing utilities
#include <ATen/DLConvertor.h>
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// Target API keyword: torch.from_dlpack

// Convert a tensor to DLPack and back using the C++ API.
static torch::Tensor roundtrip_dlpack(const torch::Tensor &tensor)
{
    DLManagedTensor *dlpack = at::toDLPack(tensor);
    torch::Tensor output = at::fromDLPack(dlpack);
    return output;
}

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
        
        // Convert tensor to DLPack format and back
        torch::Tensor output_tensor = roundtrip_dlpack(input_tensor);
        
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
            torch::Tensor another_output = roundtrip_dlpack(another_tensor);
            
            // Try operations on the converted tensor
            if (another_output.numel() > 0) {
                auto sum = another_output.sum();
                auto mean = another_output.mean();
                (void)sum;
                (void)mean;
            }
        }
        
        // Test with empty tensor if we have enough data
        if (offset + 1 < Size) {
            torch::Tensor empty_tensor = torch::empty({0});
            torch::Tensor empty_output = roundtrip_dlpack(empty_tensor);
            (void)empty_output;
        }
        
        // Test with scalar tensor
        if (offset + 1 < Size) {
            torch::Tensor scalar_tensor = torch::tensor(3.14);
            torch::Tensor scalar_output = roundtrip_dlpack(scalar_tensor);
            (void)scalar_output;
        }
        
        // Test with boolean tensor
        if (offset + 1 < Size) {
            torch::Tensor bool_tensor = torch::tensor(true);
            torch::Tensor bool_output = roundtrip_dlpack(bool_tensor);
            (void)bool_output;
        }
        
        // Test with complex tensor if we have enough data
        if (offset + 2 < Size) {
            torch::Tensor complex_tensor = torch::complex(
                torch::ones({2, 2}), 
                torch::ones({2, 2})
            );
            torch::Tensor complex_output = roundtrip_dlpack(complex_tensor);
            (void)complex_output;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
