#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }

        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (input_tensor.numel() == 0) {
            return 0;
        }
        
        if (input_tensor.dim() < 2) {
            auto shape = input_tensor.sizes().vec();
            while (shape.size() < 2) {
                shape.push_back(1);
            }
            input_tensor = input_tensor.reshape(shape);
        }
        
        auto last_two_dims = input_tensor.sizes().slice(-2);
        if (last_two_dims[0] != last_two_dims[1]) {
            int64_t min_dim = std::min(last_two_dims[0], last_two_dims[1]);
            auto shape = input_tensor.sizes().vec();
            shape[shape.size()-2] = min_dim;
            shape[shape.size()-1] = min_dim;
            input_tensor = input_tensor.slice(-2, 0, min_dim).slice(-1, 0, min_dim);
        }
        
        if (input_tensor.dtype() == torch::kBool || 
            input_tensor.dtype() == torch::kInt8 || 
            input_tensor.dtype() == torch::kUInt8 || 
            input_tensor.dtype() == torch::kInt16 || 
            input_tensor.dtype() == torch::kInt32 || 
            input_tensor.dtype() == torch::kInt64) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        auto result = torch::matrix_exp(input_tensor);
        
        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (input_tensor2.numel() > 0 && input_tensor2.dim() >= 2) {
                auto last_two_dims2 = input_tensor2.sizes().slice(-2);
                if (last_two_dims2[0] == last_two_dims2[1]) {
                    if (input_tensor2.dtype() == torch::kBool || 
                        input_tensor2.dtype() == torch::kInt8 || 
                        input_tensor2.dtype() == torch::kUInt8 || 
                        input_tensor2.dtype() == torch::kInt16 || 
                        input_tensor2.dtype() == torch::kInt32 || 
                        input_tensor2.dtype() == torch::kInt64) {
                        input_tensor2 = input_tensor2.to(torch::kFloat);
                    }
                    
                    auto result2 = torch::matrix_exp(input_tensor2);
                }
            }
        }
        
        auto zero_tensor = torch::zeros_like(input_tensor);
        auto zero_result = torch::matrix_exp(zero_tensor);
        
        auto identity_tensor = torch::eye(input_tensor.size(-1), input_tensor.options());
        if (input_tensor.dim() > 2) {
            std::vector<int64_t> batch_shape(input_tensor.sizes().begin(), input_tensor.sizes().end() - 2);
            batch_shape.push_back(input_tensor.size(-1));
            batch_shape.push_back(input_tensor.size(-1));
            identity_tensor = identity_tensor.expand(batch_shape);
        }
        auto identity_result = torch::matrix_exp(identity_tensor);
        
        auto large_tensor = input_tensor * 100.0;
        auto large_result = torch::matrix_exp(large_tensor);
        
        auto small_tensor = input_tensor * 0.01;
        auto small_result = torch::matrix_exp(small_tensor);
        
        auto negative_tensor = -input_tensor;
        auto negative_result = torch::matrix_exp(negative_tensor);
        
        if (input_tensor.dtype() == torch::kComplexFloat || input_tensor.dtype() == torch::kComplexDouble) {
            auto complex_result = torch::matrix_exp(input_tensor);
        }
        
        if (input_tensor.dtype() == torch::kFloat || input_tensor.dtype() == torch::kDouble) {
            auto complex_input = torch::complex(input_tensor, torch::zeros_like(input_tensor));
            auto complex_result = torch::matrix_exp(complex_input);
        }
        
        auto inf_tensor = torch::full_like(input_tensor, std::numeric_limits<float>::infinity());
        auto inf_result = torch::matrix_exp(inf_tensor);
        
        auto nan_tensor = torch::full_like(input_tensor, std::numeric_limits<float>::quiet_NaN());
        auto nan_result = torch::matrix_exp(nan_tensor);
        
        if (input_tensor.size(-1) <= 8) {
            auto singular_tensor = input_tensor.clone();
            singular_tensor.select(-1, 0).copy_(singular_tensor.select(-1, 1));
            auto singular_result = torch::matrix_exp(singular_tensor);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}