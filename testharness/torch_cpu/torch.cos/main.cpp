#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

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
        
        // Force computation by accessing the result
        if (result.defined() && result.numel() > 0) {
            // Use sum() instead of item() to handle multi-element tensors
            volatile auto val = result.sum().item<float>();
            (void)val;
        }
        
        // Try some edge cases with specific values if we have more data
        if (offset + 1 < Size) {
            // Create tensors with special values using float type
            auto options = torch::TensorOptions().dtype(torch::kFloat);
            auto shape = input_tensor.sizes().vec();
            if (shape.empty()) {
                shape.push_back(1);
            }
            
            // Create tensors with special values
            torch::Tensor inf_tensor = torch::full(shape, std::numeric_limits<float>::infinity(), options);
            torch::Tensor neg_inf_tensor = torch::full(shape, -std::numeric_limits<float>::infinity(), options);
            torch::Tensor nan_tensor = torch::full(shape, std::numeric_limits<float>::quiet_NaN(), options);
            
            // Apply cos to these special tensors
            torch::Tensor inf_result = torch::cos(inf_tensor);
            torch::Tensor neg_inf_result = torch::cos(neg_inf_tensor);
            torch::Tensor nan_result = torch::cos(nan_tensor);
            
            // Force computation
            volatile auto v1 = inf_result.sum().item<float>();
            volatile auto v2 = neg_inf_result.sum().item<float>();
            volatile auto v3 = nan_result.sum().item<float>();
            (void)v1; (void)v2; (void)v3;
        }
        
        // Try with different tensor options if we have more data
        if (offset + 2 < Size) {
            uint8_t option_selector = Data[offset++];
            
            // Test with non-contiguous tensor
            if (option_selector % 4 == 0 && input_tensor.dim() > 1 && input_tensor.numel() > 1) {
                try {
                    auto non_contig = input_tensor.transpose(0, input_tensor.dim() - 1);
                    if (!non_contig.is_contiguous()) {
                        torch::Tensor non_contig_result = torch::cos(non_contig);
                        volatile auto val = non_contig_result.sum().item<float>();
                        (void)val;
                    }
                } catch (...) {
                    // Transpose might fail for certain shapes
                }
            }
            
            // Test with zero-sized dimensions
            if (option_selector % 4 == 1 && input_tensor.dim() > 0) {
                std::vector<int64_t> zero_shape;
                for (int i = 0; i < input_tensor.dim(); i++) {
                    zero_shape.push_back(i == 0 ? 0 : input_tensor.size(i));
                }
                if (!zero_shape.empty()) {
                    auto options = torch::TensorOptions().dtype(torch::kFloat);
                    torch::Tensor zero_tensor = torch::empty(zero_shape, options);
                    torch::Tensor zero_result = torch::cos(zero_tensor);
                }
            }
            
            // Test with different dtype
            if (option_selector % 4 == 2 && offset < Size) {
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
                        // Force computation
                        if (converted_result.numel() > 0) {
                            converted_result.sum();
                        }
                    } catch (...) {
                        // Conversion might fail for some dtypes
                    }
                }
            }
            
            // Test with requires_grad
            if (option_selector % 4 == 3) {
                try {
                    auto float_tensor = input_tensor.to(torch::kFloat).detach();
                    float_tensor = float_tensor.set_requires_grad(true);
                    torch::Tensor grad_result = torch::cos(float_tensor);
                    
                    // Try to compute gradients if tensor is non-empty
                    if (grad_result.numel() > 0) {
                        grad_result.sum().backward();
                        // Access gradient to ensure computation
                        if (float_tensor.grad().defined()) {
                            volatile auto val = float_tensor.grad().sum().item<float>();
                            (void)val;
                        }
                    }
                } catch (...) {
                    // Gradient computation might fail
                }
            }
        }
        
        // Test in-place variant torch::cos_
        if (offset < Size && Data[offset] % 2 == 0) {
            try {
                // cos_ requires floating point tensor
                torch::Tensor inplace_tensor = input_tensor.to(torch::kFloat).clone();
                torch::cos_(inplace_tensor);
                if (inplace_tensor.numel() > 0) {
                    volatile auto val = inplace_tensor.sum().item<float>();
                    (void)val;
                }
            } catch (...) {
                // In-place operation might fail
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}