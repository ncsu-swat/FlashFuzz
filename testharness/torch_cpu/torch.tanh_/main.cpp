#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make a copy of the original tensor for verification
        torch::Tensor original = tensor.clone();
        
        // Apply tanh_ in-place operation
        tensor.tanh_();
        
        // Verify the operation worked correctly by comparing with non-in-place version
        torch::Tensor expected = torch::tanh(original);
        
        // Check if the in-place operation produced the same result as the non-in-place version
        if (!torch::allclose(tensor, expected)) {
            throw std::runtime_error("tanh_ produced different results than tanh");
        }
        
        // Try with different tensor options if we have more data
        if (offset + 2 < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Apply tanh_ to the second tensor
            tensor2.tanh_();
            
            // Try with a view of the tensor to test in-place ops on views
            if (tensor2.numel() > 0 && tensor2.dim() > 0) {
                torch::Tensor view = tensor2.slice(0, 0, tensor2.size(0));
                view.tanh_();
            }
        }
        
        // Try with a scalar tensor (0-dim)
        if (offset + 2 < Size) {
            torch::Tensor scalar_tensor = torch::tensor(Data[offset] / 255.0);
            scalar_tensor.tanh_();
        }
        
        // Try with empty tensor
        torch::Tensor empty_tensor = torch::empty({0});
        empty_tensor.tanh_();
        
        // Try with tensors of different dtypes
        if (offset + 2 < Size) {
            // Float tensor
            torch::Tensor float_tensor = torch::tensor({{Data[offset] / 255.0, (Data[offset+1] % 100) / 50.0}}, 
                                                      torch::TensorOptions().dtype(torch::kFloat));
            float_tensor.tanh_();
            
            // Double tensor
            torch::Tensor double_tensor = torch::tensor({{Data[offset] / 255.0, (Data[offset+1] % 100) / 50.0}}, 
                                                       torch::TensorOptions().dtype(torch::kDouble));
            double_tensor.tanh_();
            
            // Half tensor (if supported)
            try {
                torch::Tensor half_tensor = torch::tensor({{Data[offset] / 255.0, (Data[offset+1] % 100) / 50.0}}, 
                                                         torch::TensorOptions().dtype(torch::kHalf));
                half_tensor.tanh_();
            } catch (const std::exception&) {
                // Half precision might not be supported on all platforms
            }
            
            // Complex tensor using c10::complex
            try {
                torch::Tensor complex_tensor = torch::tensor({{c10::complex<float>(Data[offset] / 255.0, Data[offset+1] / 255.0)}}, 
                                                           torch::TensorOptions().dtype(torch::kComplexFloat));
                complex_tensor.tanh_();
            } catch (const std::exception&) {
                // Complex tanh might not be supported
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}