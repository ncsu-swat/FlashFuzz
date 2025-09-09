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
        auto tau_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        bool left = (Data[offset] % 2) == 0;
        offset++;
        
        if (offset >= Size) {
            return 0;
        }
        
        bool transpose = (Data[offset] % 2) == 0;
        offset++;
        
        if (input_tensor.numel() == 0 || tau_tensor.numel() == 0) {
            return 0;
        }
        
        if (input_tensor.dim() < 2) {
            input_tensor = input_tensor.unsqueeze(0);
            if (input_tensor.dim() < 2) {
                input_tensor = input_tensor.unsqueeze(-1);
            }
        }
        
        if (tau_tensor.dim() == 0) {
            tau_tensor = tau_tensor.unsqueeze(0);
        }
        
        auto input_sizes = input_tensor.sizes();
        auto tau_sizes = tau_tensor.sizes();
        
        int64_t m = input_sizes[input_sizes.size() - 2];
        int64_t n = input_sizes[input_sizes.size() - 1];
        int64_t k = std::min(m, n);
        
        if (tau_tensor.numel() < k && k > 0) {
            tau_tensor = tau_tensor.expand({k});
        }
        
        if (input_tensor.dtype() != torch::kFloat && input_tensor.dtype() != torch::kDouble && 
            input_tensor.dtype() != torch::kComplexFloat && input_tensor.dtype() != torch::kComplexDouble) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        if (tau_tensor.dtype() != input_tensor.dtype()) {
            tau_tensor = tau_tensor.to(input_tensor.dtype());
        }
        
        torch::Tensor other_tensor;
        if (offset < Size) {
            other_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (other_tensor.numel() == 0) {
                other_tensor = torch::ones({m, n}, input_tensor.options());
            }
            
            if (other_tensor.dim() < 2) {
                other_tensor = other_tensor.unsqueeze(0);
                if (other_tensor.dim() < 2) {
                    other_tensor = other_tensor.unsqueeze(-1);
                }
            }
            
            if (other_tensor.dtype() != input_tensor.dtype()) {
                other_tensor = other_tensor.to(input_tensor.dtype());
            }
            
            auto other_sizes = other_tensor.sizes();
            int64_t other_m = other_sizes[other_sizes.size() - 2];
            int64_t other_n = other_sizes[other_sizes.size() - 1];
            
            if (left) {
                if (other_m != m) {
                    if (other_m < m) {
                        other_tensor = torch::cat({other_tensor, torch::zeros({m - other_m, other_n}, other_tensor.options())}, 0);
                    } else {
                        other_tensor = other_tensor.narrow(0, 0, m);
                    }
                }
            } else {
                if (other_n != n) {
                    if (other_n < n) {
                        other_tensor = torch::cat({other_tensor, torch::zeros({other_m, n - other_n}, other_tensor.options())}, 1);
                    } else {
                        other_tensor = other_tensor.narrow(1, 0, n);
                    }
                }
            }
        } else {
            other_tensor = torch::ones({m, n}, input_tensor.options());
        }
        
        auto result = torch::ormqr(input_tensor, tau_tensor, other_tensor, left, transpose);
        
        if (result.numel() > 1000000) {
            return 0;
        }
        
        auto sum_result = torch::sum(result);
        if (torch::isnan(sum_result).item<bool>() || torch::isinf(sum_result).item<bool>()) {
            return 0;
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}