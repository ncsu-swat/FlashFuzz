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
        
        bool enable_inference_mode = Data[offset++] % 2;
        
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        torch::Tensor result;
        
        if (enable_inference_mode) {
            torch::InferenceMode guard(true);
            result = input_tensor * 2.0;
            result = torch::relu(result);
            result = result.sum();
            
            torch::Tensor another_tensor = torch::randn({3, 3});
            torch::Tensor matmul_result = torch::mm(another_tensor, another_tensor.t());
            
            if (input_tensor.dim() >= 2 && input_tensor.size(0) > 0 && input_tensor.size(1) > 0) {
                torch::Tensor reshaped = input_tensor.view({-1});
                if (reshaped.numel() >= 9) {
                    torch::Tensor subset = reshaped.slice(0, 0, 9).view({3, 3});
                    torch::Tensor combined = subset + matmul_result;
                    result = combined.sum();
                }
            }
        } else {
            result = input_tensor * 3.0;
            result = torch::sigmoid(result);
            result = result.mean();
        }
        
        {
            torch::InferenceMode nested_guard(false);
            torch::Tensor temp = torch::ones_like(input_tensor);
            temp.requires_grad_(true);
            torch::Tensor grad_result = temp * input_tensor;
            
            {
                torch::InferenceMode double_nested(true);
                torch::Tensor no_grad_ops = grad_result.detach();
                no_grad_ops = torch::tanh(no_grad_ops);
                result = result + no_grad_ops.sum();
            }
        }
        
        torch::InferenceMode::is_enabled();
        
        {
            torch::InferenceMode toggle_guard(enable_inference_mode);
            torch::Tensor conditional_tensor = enable_inference_mode ? 
                torch::zeros({2, 2}) : torch::ones({2, 2});
            conditional_tensor = conditional_tensor.to(input_tensor.dtype());
            
            if (input_tensor.numel() >= 4) {
                torch::Tensor flat_input = input_tensor.flatten();
                torch::Tensor subset = flat_input.slice(0, 0, 4).view({2, 2});
                torch::Tensor element_wise = subset * conditional_tensor;
                result = result + element_wise.sum();
            }
        }
        
        if (offset < Size) {
            bool nested_mode = Data[offset % Size] % 2;
            torch::InferenceMode outer_guard(nested_mode);
            
            torch::Tensor outer_tensor = torch::randn({5, 5});
            
            {
                torch::InferenceMode inner_guard(!nested_mode);
                torch::Tensor inner_tensor = torch::eye(5);
                torch::Tensor combined = outer_tensor + inner_tensor;
                
                {
                    torch::InferenceMode deepest_guard(nested_mode);
                    combined = torch::softmax(combined, 1);
                    result = result + combined.trace();
                }
            }
        }
        
        result.item<double>();
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}