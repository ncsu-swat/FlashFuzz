#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }
    
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create first tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // bitwise_or only works on integral types (bool, int, etc.), not floating point
        // Convert to an integral type based on fuzzer data
        torch::ScalarType integral_dtype;
        if (offset < Size) {
            uint8_t dtype_choice = Data[offset++] % 5;
            switch (dtype_choice) {
                case 0: integral_dtype = torch::kBool; break;
                case 1: integral_dtype = torch::kUInt8; break;
                case 2: integral_dtype = torch::kInt32; break;
                case 3: integral_dtype = torch::kInt64; break;
                default: integral_dtype = torch::kInt16; break;
            }
        } else {
            integral_dtype = torch::kInt32;
        }
        
        // Convert tensor1 to integral type
        tensor1 = tensor1.to(integral_dtype);
        
        // Create second tensor if there's data left
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            tensor2 = tensor2.to(integral_dtype);
        } else {
            // If no data left, use the same tensor for both inputs
            tensor2 = tensor1.clone();
        }
        
        // Try scalar version if there's at least one more byte
        if (offset < Size) {
            int64_t scalar_value = static_cast<int64_t>(Data[offset++]);
            
            try {
                // Test tensor | scalar
                torch::Tensor result1 = torch::bitwise_or(tensor1, scalar_value);
                
                // Test scalar | tensor (using Scalar explicitly)
                torch::Tensor result2 = torch::bitwise_or(torch::scalar_tensor(scalar_value, tensor1.options()), tensor1);
            } catch (...) {
                // Scalar operations might fail for some configurations
            }
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
            try {
                torch::Tensor tensor_copy = tensor1.clone();
                tensor_copy.bitwise_or_(tensor2);
            } catch (...) {
                // In-place might fail due to shape/type issues
            }
        }
        
        // Try out= version
        if (offset < Size && Data[offset++] % 2 == 0) {
            try {
                torch::Tensor out_tensor;
                
                // Sometimes create output tensor with same shape as expected result
                if (offset < Size && Data[offset % Size] % 2 == 0) {
                    out_tensor = torch::empty_like(result3);
                } else {
                    // Sometimes create output tensor with different shape to test resizing
                    out_tensor = torch::empty(0, tensor1.options());
                }
                
                torch::bitwise_or_out(out_tensor, tensor1, tensor2);
            } catch (...) {
                // Out version might fail
            }
        }
        
        // Try with different integral dtypes
        if (offset + 1 < Size) {
            try {
                uint8_t dtype_selector = Data[offset++] % 5;
                torch::ScalarType new_dtype;
                switch (dtype_selector) {
                    case 0: new_dtype = torch::kBool; break;
                    case 1: new_dtype = torch::kUInt8; break;
                    case 2: new_dtype = torch::kInt32; break;
                    case 3: new_dtype = torch::kInt64; break;
                    default: new_dtype = torch::kInt16; break;
                }
                
                // Convert tensors to the new dtype
                torch::Tensor converted1 = tensor1.to(new_dtype);
                torch::Tensor converted2 = tensor2.to(new_dtype);
                
                // Perform bitwise_or with converted tensors
                torch::Tensor converted_result = torch::bitwise_or(converted1, converted2);
            } catch (...) {
                // Conversion might fail for some dtypes, that's fine
            }
        }
        
        // Test with boolean tensors specifically (common use case)
        if (offset < Size && Data[offset++] % 3 == 0) {
            try {
                torch::Tensor bool1 = tensor1.to(torch::kBool);
                torch::Tensor bool2 = tensor2.to(torch::kBool);
                torch::Tensor bool_result = torch::bitwise_or(bool1, bool2);
            } catch (...) {
                // Boolean conversion might fail
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}