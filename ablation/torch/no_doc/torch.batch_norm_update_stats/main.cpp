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

        auto input = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        auto running_mean = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        auto running_var = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset + 8 > Size) {
            return 0;
        }
        
        double momentum;
        std::memcpy(&momentum, Data + offset, sizeof(double));
        offset += sizeof(double);
        
        if (input.dim() < 2) {
            return 0;
        }
        
        int64_t num_features = input.size(1);
        
        if (running_mean.numel() == 0) {
            running_mean = torch::zeros({num_features}, input.options());
        } else if (running_mean.numel() != num_features) {
            running_mean = running_mean.view({-1}).slice(0, 0, std::min(running_mean.numel(), num_features));
            if (running_mean.numel() < num_features) {
                auto padding = torch::zeros({num_features - running_mean.numel()}, input.options());
                running_mean = torch::cat({running_mean, padding});
            }
            running_mean = running_mean.slice(0, 0, num_features);
        }
        
        if (running_var.numel() == 0) {
            running_var = torch::ones({num_features}, input.options());
        } else if (running_var.numel() != num_features) {
            running_var = running_var.view({-1}).slice(0, 0, std::min(running_var.numel(), num_features));
            if (running_var.numel() < num_features) {
                auto padding = torch::ones({num_features - running_var.numel()}, input.options());
                running_var = torch::cat({running_var, padding});
            }
            running_var = running_var.slice(0, 0, num_features);
        }
        
        running_mean = running_mean.to(input.dtype()).contiguous();
        running_var = running_var.to(input.dtype()).contiguous();
        
        if (input.dtype() == torch::kComplexFloat || input.dtype() == torch::kComplexDouble) {
            input = input.real();
        }
        
        if (running_mean.dtype() == torch::kComplexFloat || running_mean.dtype() == torch::kComplexDouble) {
            running_mean = running_mean.real();
        }
        
        if (running_var.dtype() == torch::kComplexFloat || running_var.dtype() == torch::kComplexDouble) {
            running_var = running_var.real();
        }
        
        if (input.dtype() == torch::kBool) {
            input = input.to(torch::kFloat);
        }
        if (running_mean.dtype() == torch::kBool) {
            running_mean = running_mean.to(torch::kFloat);
        }
        if (running_var.dtype() == torch::kBool) {
            running_var = running_var.to(torch::kFloat);
        }
        
        if (input.dtype() == torch::kInt8 || input.dtype() == torch::kUInt8 || 
            input.dtype() == torch::kInt16 || input.dtype() == torch::kInt32 || 
            input.dtype() == torch::kInt64) {
            input = input.to(torch::kFloat);
        }
        if (running_mean.dtype() == torch::kInt8 || running_mean.dtype() == torch::kUInt8 || 
            running_mean.dtype() == torch::kInt16 || running_mean.dtype() == torch::kInt32 || 
            running_mean.dtype() == torch::kInt64) {
            running_mean = running_mean.to(torch::kFloat);
        }
        if (running_var.dtype() == torch::kInt8 || running_var.dtype() == torch::kUInt8 || 
            running_var.dtype() == torch::kInt16 || running_var.dtype() == torch::kInt32 || 
            running_var.dtype() == torch::kInt64) {
            running_var = running_var.to(torch::kFloat);
        }
        
        auto result = torch::batch_norm_update_stats(input, running_mean, running_var, momentum);
        
        if (result.numel() > 0) {
            auto sum_result = torch::sum(result);
            if (sum_result.numel() > 0) {
                volatile auto item_val = sum_result.item<float>();
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