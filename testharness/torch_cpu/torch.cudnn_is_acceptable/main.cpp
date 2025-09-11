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
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Call cudnn_is_acceptable on the tensor
        bool is_acceptable = torch::cudnn_is_acceptable(tensor);
        
        // Try with different tensor types and shapes if we have more data
        if (offset + 2 < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            bool is_acceptable2 = torch::cudnn_is_acceptable(tensor2);
        }
        
        // Try with a tensor that has special properties
        if (offset + 2 < Size) {
            // Create a tensor with potentially challenging properties
            uint8_t dtype_selector = Data[offset++] % 12;
            torch::ScalarType dtype;
            
            // Select different dtypes to test
            switch (dtype_selector) {
                case 0: dtype = torch::kFloat; break;
                case 1: dtype = torch::kDouble; break;
                case 2: dtype = torch::kHalf; break;
                case 3: dtype = torch::kBFloat16; break;
                case 4: dtype = torch::kInt8; break;
                case 5: dtype = torch::kInt16; break;
                case 6: dtype = torch::kInt32; break;
                case 7: dtype = torch::kInt64; break;
                case 8: dtype = torch::kUInt8; break;
                case 9: dtype = torch::kBool; break;
                case 10: dtype = torch::kComplexFloat; break;
                case 11: dtype = torch::kComplexDouble; break;
                default: dtype = torch::kFloat;
            }
            
            // Create tensors with various shapes
            std::vector<int64_t> shape;
            uint8_t rank = (offset < Size) ? Data[offset++] % 5 : 2;
            
            for (uint8_t i = 0; i < rank; i++) {
                int64_t dim = (offset < Size) ? static_cast<int64_t>(Data[offset++]) : 1;
                shape.push_back(dim);
            }
            
            // Create tensor with the specified properties
            torch::Tensor special_tensor = torch::empty(shape, torch::TensorOptions().dtype(dtype));
            
            // Test cudnn_is_acceptable with this tensor
            bool is_special_acceptable = torch::cudnn_is_acceptable(special_tensor);
            
            // Try with non-contiguous tensor if possible
            if (rank >= 2 && shape[0] > 1 && shape[1] > 1) {
                torch::Tensor non_contiguous = special_tensor.transpose(0, 1);
                bool is_non_contiguous_acceptable = torch::cudnn_is_acceptable(non_contiguous);
            }
        }
        
        // Try with empty tensor
        torch::Tensor empty_tensor = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat));
        bool is_empty_acceptable = torch::cudnn_is_acceptable(empty_tensor);
        
        // Try with scalar tensor
        torch::Tensor scalar_tensor = torch::tensor(1.0f);
        bool is_scalar_acceptable = torch::cudnn_is_acceptable(scalar_tensor);
        
        // Try with very large tensor if we have enough data
        if (offset < Size) {
            int64_t large_dim = std::max<int64_t>(1, Data[offset++]);
            try {
                torch::Tensor large_tensor = torch::empty({large_dim, large_dim}, 
                                                         torch::TensorOptions().dtype(torch::kFloat));
                bool is_large_acceptable = torch::cudnn_is_acceptable(large_tensor);
            } catch (const std::exception& e) {
                // Catch and ignore memory allocation errors for very large tensors
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
