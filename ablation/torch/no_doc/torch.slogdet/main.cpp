#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 3) {
            return 0;
        }
        
        auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (input_tensor.dim() < 2) {
            auto shape = input_tensor.sizes().vec();
            while (shape.size() < 2) {
                shape.push_back(1);
            }
            input_tensor = input_tensor.reshape(shape);
        }
        
        int64_t last_dim = input_tensor.size(-1);
        int64_t second_last_dim = input_tensor.size(-2);
        
        if (last_dim != second_last_dim) {
            int64_t min_dim = std::min(last_dim, second_last_dim);
            auto shape = input_tensor.sizes().vec();
            shape[shape.size() - 1] = min_dim;
            shape[shape.size() - 2] = min_dim;
            input_tensor = input_tensor.narrow(-1, 0, min_dim).narrow(-2, 0, min_dim);
        }
        
        if (input_tensor.dtype() == torch::kBool || 
            input_tensor.dtype() == torch::kInt8 || 
            input_tensor.dtype() == torch::kUInt8 || 
            input_tensor.dtype() == torch::kInt16 || 
            input_tensor.dtype() == torch::kInt32 || 
            input_tensor.dtype() == torch::kInt64) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        auto result = torch::slogdet(input_tensor);
        auto sign = std::get<0>(result);
        auto logabsdet = std::get<1>(result);
        
        if (sign.numel() > 0) {
            sign.sum();
        }
        if (logabsdet.numel() > 0) {
            logabsdet.sum();
        }
        
        if (offset < Size) {
            auto input_tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (input_tensor2.dim() >= 2) {
                int64_t last_dim2 = input_tensor2.size(-1);
                int64_t second_last_dim2 = input_tensor2.size(-2);
                
                if (last_dim2 == second_last_dim2) {
                    if (input_tensor2.dtype() == torch::kBool || 
                        input_tensor2.dtype() == torch::kInt8 || 
                        input_tensor2.dtype() == torch::kUInt8 || 
                        input_tensor2.dtype() == torch::kInt16 || 
                        input_tensor2.dtype() == torch::kInt32 || 
                        input_tensor2.dtype() == torch::kInt64) {
                        input_tensor2 = input_tensor2.to(torch::kFloat);
                    }
                    
                    auto result2 = torch::slogdet(input_tensor2);
                    auto sign2 = std::get<0>(result2);
                    auto logabsdet2 = std::get<1>(result2);
                    
                    if (sign2.numel() > 0) {
                        sign2.sum();
                    }
                    if (logabsdet2.numel() > 0) {
                        logabsdet2.sum();
                    }
                }
            }
        }
        
        auto zero_tensor = torch::zeros({2, 2});
        auto zero_result = torch::slogdet(zero_tensor);
        
        auto identity_tensor = torch::eye(3);
        auto identity_result = torch::slogdet(identity_tensor);
        
        auto singular_tensor = torch::tensor({{1.0, 2.0}, {2.0, 4.0}});
        auto singular_result = torch::slogdet(singular_tensor);
        
        auto large_tensor = torch::randn({100, 100}) * 1000.0;
        auto large_result = torch::slogdet(large_tensor);
        
        auto small_tensor = torch::randn({100, 100}) * 1e-10;
        auto small_result = torch::slogdet(small_tensor);
        
        auto batch_tensor = torch::randn({5, 3, 3});
        auto batch_result = torch::slogdet(batch_tensor);
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}