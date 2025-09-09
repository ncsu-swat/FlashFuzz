#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        bool inference_mode_enabled = Data[offset++] % 2;
        
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        x.requires_grad_(true);
        
        torch::Tensor y;
        
        if (inference_mode_enabled) {
            torch::InferenceMode guard(true);
            y = x * x + x.sin() + x.cos();
            y = y.view({-1});
            y = y.transpose(0, 0);
            y = y.clone();
            y = y.detach();
            
            if (offset < Size) {
                torch::Tensor z = fuzzer_utils::createTensor(Data, Size, offset);
                z.requires_grad_(true);
                y = y + z.sum();
            }
            
            auto version_check = [&]() {
                try {
                    auto version = y._version();
                } catch (...) {
                }
            };
            version_check();
            
        } else {
            y = x * x + x.sin() + x.cos();
            y = y.view({-1});
            y = y.transpose(0, 0);
            y = y.clone();
            y = y.detach();
            
            if (offset < Size) {
                torch::Tensor z = fuzzer_utils::createTensor(Data, Size, offset);
                z.requires_grad_(true);
                y = y + z.sum();
            }
        }
        
        bool requires_grad_result = y.requires_grad();
        
        {
            torch::InferenceMode nested_guard(true);
            torch::Tensor nested_result = y * 2;
            nested_result = nested_result.abs();
            nested_result = nested_result.sqrt();
            
            {
                torch::InferenceMode double_nested(false);
                torch::Tensor double_nested_result = nested_result + 1;
                double_nested_result.requires_grad();
            }
        }
        
        if (offset < Size) {
            torch::InferenceMode conditional_guard(Data[offset % Size] % 2);
            torch::Tensor conditional_result = x.pow(2);
            conditional_result = conditional_result.mean();
        }
        
        auto lambda_with_inference = [&]() {
            torch::InferenceMode lambda_guard(true);
            return x.sum() * x.prod();
        };
        auto lambda_result = lambda_with_inference();
        
        torch::Tensor final_computation;
        {
            torch::InferenceMode final_guard(inference_mode_enabled);
            final_computation = y + lambda_result;
            final_computation = final_computation.flatten();
            
            if (final_computation.numel() > 0) {
                final_computation = final_computation[0];
            }
        }
        
        auto thread_local_test = [&]() {
            torch::InferenceMode thread_guard(true);
            torch::Tensor thread_result = x.clone();
            thread_result = thread_result * 3;
            return thread_result.requires_grad();
        };
        bool thread_requires_grad = thread_local_test();
        
        if (Size > 10) {
            torch::InferenceMode stress_guard(Data[Size-1] % 2);
            for (int i = 0; i < 5; ++i) {
                torch::Tensor stress_tensor = x.clone();
                stress_tensor = stress_tensor + i;
                stress_tensor = stress_tensor.relu();
                stress_tensor.requires_grad();
            }
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}