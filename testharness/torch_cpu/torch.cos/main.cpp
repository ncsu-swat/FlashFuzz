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
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.cos operation
        torch::Tensor result = torch::cos(input_tensor);
        
        // Try to access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            auto item = result.item();
        }
        
        // Try some edge cases with specific values if we have more data
        if (offset + 1 < Size) {
            // Create a tensor with extreme values
            auto options = torch::TensorOptions().dtype(input_tensor.dtype());
            
            // Create tensors with special values
            torch::Tensor inf_tensor = torch::full_like(input_tensor, std::numeric_limits<float>::infinity());
            torch::Tensor neg_inf_tensor = torch::full_like(input_tensor, -std::numeric_limits<float>::infinity());
            torch::Tensor nan_tensor = torch::full_like(input_tensor, std::numeric_limits<float>::quiet_NaN());
            
            // Apply cos to these special tensors
            torch::Tensor inf_result = torch::cos(inf_tensor);
            torch::Tensor neg_inf_result = torch::cos(neg_inf_tensor);
            torch::Tensor nan_result = torch::cos(nan_tensor);
        }
        
        // Try with different tensor options if we have more data
        if (offset + 2 < Size) {
            uint8_t option_selector = Data[offset++];
            
            // Test with non-contiguous tensor
            if (option_selector % 4 == 0 && input_tensor.dim() > 0 && input_tensor.numel() > 1) {
                auto non_contig = input_tensor.transpose(0, input_tensor.dim() - 1);
                if (!non_contig.is_contiguous()) {
                    torch::Tensor non_contig_result = torch::cos(non_contig);
                }
            }
            
            // Test with zero-sized dimensions
            if (option_selector % 4 == 1) {
                std::vector<int64_t> zero_shape;
                for (int i = 0; i < input_tensor.dim(); i++) {
                    zero_shape.push_back(i == 0 ? 0 : input_tensor.size(i));
                }
                if (!zero_shape.empty()) {
                    auto options = torch::TensorOptions().dtype(input_tensor.dtype());
                    torch::Tensor zero_tensor = torch::empty(zero_shape, options);
                    torch::Tensor zero_result = torch::cos(zero_tensor);
                }
            }
            
            // Test with different dtype
            if (option_selector % 4 == 2) {
                auto dtype_selector = Data[offset++] % 3;
                torch::ScalarType target_dtype;
                
                switch (dtype_selector) {
                    case 0: target_dtype = torch::kFloat; break;
                    case 1: target_dtype = torch::kDouble; break;
                    case 2: target_dtype = torch::kComplexFloat; break;
                    default: target_dtype = torch::kFloat;
                }
                
                if (input_tensor.scalar_type() != target_dtype) {
                    try {
                        auto converted = input_tensor.to(target_dtype);
                        torch::Tensor converted_result = torch::cos(converted);
                    } catch (...) {
                        // Conversion might fail for some dtypes, that's ok
                    }
                }
            }
            
            // Test with requires_grad
            if (option_selector % 4 == 3) {
                try {
                    auto float_tensor = input_tensor.to(torch::kFloat);
                    float_tensor = float_tensor.set_requires_grad(true);
                    torch::Tensor grad_result = torch::cos(float_tensor);
                    
                    // Try to compute gradients if tensor is non-empty
                    if (grad_result.numel() > 0) {
                        grad_result.sum().backward();
                    }
                } catch (...) {
                    // Gradient computation might fail, that's ok
                }
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
