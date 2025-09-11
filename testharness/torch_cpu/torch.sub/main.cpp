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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create first input tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor or scalar
        bool use_scalar = false;
        if (offset < Size) {
            use_scalar = Data[offset++] % 2 == 0;
        }
        
        // Test different variants of sub
        if (use_scalar) {
            // Use a scalar value for subtraction
            double scalar_value = 0.0;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&scalar_value, Data + offset, sizeof(double));
                offset += sizeof(double);
            }
            
            // Test torch::sub with scalar
            torch::Tensor result1 = torch::sub(tensor1, scalar_value);
            
            // Test tensor.sub with scalar
            torch::Tensor result2 = tensor1.sub(scalar_value);
            
            // Test in-place subtraction with scalar
            torch::Tensor tensor_copy = tensor1.clone();
            tensor_copy.sub_(scalar_value);
        } else {
            // Create second tensor for tensor-tensor subtraction
            torch::Tensor tensor2;
            if (offset < Size) {
                tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                // If we don't have enough data, create a simple tensor
                tensor2 = torch::ones_like(tensor1);
            }
            
            // Test different variants of sub operation
            try {
                // Test torch::sub with two tensors
                torch::Tensor result1 = torch::sub(tensor1, tensor2);
                
                // Test tensor.sub with another tensor
                torch::Tensor result2 = tensor1.sub(tensor2);
                
                // Test in-place subtraction
                torch::Tensor tensor_copy = tensor1.clone();
                tensor_copy.sub_(tensor2);
                
                // Test with alpha parameter
                double alpha = 1.0;
                if (offset + sizeof(double) <= Size) {
                    std::memcpy(&alpha, Data + offset, sizeof(double));
                    offset += sizeof(double);
                }
                
                // Test torch::sub with alpha
                torch::Tensor result3 = torch::sub(tensor1, tensor2, alpha);
                
                // Test tensor.sub with alpha
                torch::Tensor result4 = tensor1.sub(tensor2, alpha);
                
                // Test in-place subtraction with alpha
                torch::Tensor tensor_copy2 = tensor1.clone();
                tensor_copy2.sub_(tensor2, alpha);
            } catch (const c10::Error& e) {
                // PyTorch specific errors are expected for incompatible shapes
                return 0;
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
