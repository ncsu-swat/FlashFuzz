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
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply is_conj operation
        bool result = tensor.is_conj();
        
        // Create a conjugated tensor and test is_conj on it
        if (offset + 1 < Size) {
            // Use the next byte to decide whether to create a conjugated tensor
            uint8_t should_conjugate = Data[offset++];
            
            if (should_conjugate % 2 == 1) {
                // Create a conjugated tensor
                torch::Tensor conj_tensor = tensor.conj();
                
                // Test is_conj on the conjugated tensor
                bool conj_result = conj_tensor.is_conj();
                
                // Use the result to prevent optimization
                if (conj_result) {
                    // Test that conjugating a conjugated tensor works
                    torch::Tensor double_conj = conj_tensor.conj();
                    bool double_conj_result = double_conj.is_conj();
                    
                    // Test that conjugating a view works
                    if (!tensor.sizes().empty() && tensor.numel() > 0) {
                        torch::Tensor view_tensor = tensor.view({-1});
                        torch::Tensor conj_view = view_tensor.conj();
                        bool view_conj_result = conj_view.is_conj();
                    }
                }
            }
        }
        
        // Test is_conj with different tensor types if we have more data
        if (offset + 1 < Size) {
            uint8_t tensor_type = Data[offset++];
            
            // Create tensors of different types to test is_conj
            if (tensor_type % 5 == 0) {
                // Complex tensor
                auto options = torch::TensorOptions().dtype(torch::kComplexFloat);
                torch::Tensor complex_tensor = torch::empty({1, 1}, options);
                bool complex_is_conj = complex_tensor.is_conj();
                
                // Conjugated complex tensor
                torch::Tensor conj_complex = complex_tensor.conj();
                bool conj_complex_is_conj = conj_complex.is_conj();
            } else if (tensor_type % 5 == 1) {
                // Float tensor
                auto options = torch::TensorOptions().dtype(torch::kFloat);
                torch::Tensor float_tensor = torch::empty({1, 1}, options);
                bool float_is_conj = float_tensor.is_conj();
                
                // Conjugated float tensor
                torch::Tensor conj_float = float_tensor.conj();
                bool conj_float_is_conj = conj_float.is_conj();
            } else if (tensor_type % 5 == 2) {
                // Integer tensor
                auto options = torch::TensorOptions().dtype(torch::kInt);
                torch::Tensor int_tensor = torch::empty({1, 1}, options);
                bool int_is_conj = int_tensor.is_conj();
                
                // Conjugated integer tensor
                torch::Tensor conj_int = int_tensor.conj();
                bool conj_int_is_conj = conj_int.is_conj();
            } else if (tensor_type % 5 == 3) {
                // Boolean tensor
                auto options = torch::TensorOptions().dtype(torch::kBool);
                torch::Tensor bool_tensor = torch::empty({1, 1}, options);
                bool bool_is_conj = bool_tensor.is_conj();
                
                // Conjugated boolean tensor
                torch::Tensor conj_bool = bool_tensor.conj();
                bool conj_bool_is_conj = conj_bool.is_conj();
            } else {
                // Empty tensor
                auto options = torch::TensorOptions().dtype(torch::kFloat);
                torch::Tensor empty_tensor = torch::empty({0}, options);
                bool empty_is_conj = empty_tensor.is_conj();
                
                // Conjugated empty tensor
                torch::Tensor conj_empty = empty_tensor.conj();
                bool conj_empty_is_conj = conj_empty.is_conj();
            }
        }
        
        // Test is_conj with scalar tensor
        if (offset + 1 < Size) {
            uint8_t scalar_type = Data[offset++];
            
            if (scalar_type % 2 == 0) {
                // Create a scalar tensor
                torch::Tensor scalar_tensor = torch::tensor(1.0);
                bool scalar_is_conj = scalar_tensor.is_conj();
                
                // Conjugated scalar tensor
                torch::Tensor conj_scalar = scalar_tensor.conj();
                bool conj_scalar_is_conj = conj_scalar.is_conj();
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
