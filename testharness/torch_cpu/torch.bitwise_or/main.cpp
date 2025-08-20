#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create first tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor if there's data left
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no data left, use the same tensor for both inputs
            tensor2 = tensor1.clone();
        }
        
        // Try scalar version if there's at least one more byte
        if (offset < Size) {
            uint8_t scalar_value = Data[offset++];
            
            // Test scalar | tensor
            torch::Tensor result1 = torch::bitwise_or(scalar_value, tensor1);
            
            // Test tensor | scalar
            torch::Tensor result2 = torch::bitwise_or(tensor1, scalar_value);
        }
        
        // Try tensor | tensor version
        torch::Tensor result3;
        
        // Try different broadcasting scenarios
        if (offset < Size && Data[offset++] % 2 == 0) {
            // Try to reshape tensor2 to force broadcasting
            if (tensor2.dim() > 0 && tensor2.numel() > 0) {
                std::vector<int64_t> new_shape;
                for (int i = 0; i < tensor2.dim(); i++) {
                    if (i < tensor2.dim() - 1) {
                        new_shape.push_back(tensor2.size(i));
                    } else {
                        new_shape.push_back(1);
                    }
                }
                
                try {
                    tensor2 = tensor2.reshape(new_shape);
                } catch (...) {
                    // Reshape failed, continue with original tensor
                }
            }
        }
        
        // Perform the bitwise_or operation
        result3 = torch::bitwise_or(tensor1, tensor2);
        
        // Try in-place version
        if (offset < Size && Data[offset++] % 2 == 0) {
            torch::Tensor tensor_copy = tensor1.clone();
            tensor_copy.bitwise_or_(tensor2);
        }
        
        // Try out= version
        if (offset < Size && Data[offset++] % 2 == 0) {
            torch::Tensor out_tensor;
            
            // Sometimes create output tensor with same shape as expected result
            if (Data[offset % Size] % 2 == 0) {
                out_tensor = torch::empty_like(result3);
            } else {
                // Sometimes create output tensor with different shape to test resizing
                out_tensor = torch::empty(0, tensor1.options());
            }
            
            torch::bitwise_or_out(out_tensor, tensor1, tensor2);
        }
        
        // Try with different dtypes if there's data left
        if (offset + 1 < Size) {
            try {
                uint8_t dtype_selector = Data[offset++];
                torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
                
                // Convert tensors to the new dtype
                torch::Tensor converted1 = tensor1.to(dtype);
                torch::Tensor converted2 = tensor2.to(dtype);
                
                // Perform bitwise_or with converted tensors
                torch::Tensor converted_result = torch::bitwise_or(converted1, converted2);
            } catch (...) {
                // Conversion might fail for some dtypes, that's fine
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