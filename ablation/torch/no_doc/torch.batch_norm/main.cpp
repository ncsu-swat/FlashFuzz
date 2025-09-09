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
        
        if (input.dim() < 2) {
            return 0;
        }
        
        int64_t num_features = input.size(1);
        
        if (num_features <= 0) {
            return 0;
        }
        
        auto running_mean = torch::zeros({num_features}, input.options());
        auto running_var = torch::ones({num_features}, input.options());
        auto weight = torch::ones({num_features}, input.options());
        auto bias = torch::zeros({num_features}, input.options());
        
        uint8_t training_byte = (offset < Size) ? Data[offset++] : 0;
        bool training = (training_byte % 2) == 1;
        
        uint8_t momentum_bytes[4] = {0};
        size_t momentum_size = std::min(size_t(4), Size - offset);
        for (size_t i = 0; i < momentum_size; ++i) {
            momentum_bytes[i] = Data[offset++];
        }
        float momentum_raw;
        std::memcpy(&momentum_raw, momentum_bytes, sizeof(float));
        double momentum = static_cast<double>(momentum_raw);
        if (std::isnan(momentum) || std::isinf(momentum)) {
            momentum = 0.1;
        }
        momentum = std::abs(momentum);
        if (momentum > 1.0) {
            momentum = 1.0;
        }
        
        uint8_t eps_bytes[4] = {0};
        size_t eps_size = std::min(size_t(4), Size - offset);
        for (size_t i = 0; i < eps_size; ++i) {
            eps_bytes[i] = Data[offset++];
        }
        float eps_raw;
        std::memcpy(&eps_raw, eps_bytes, sizeof(float));
        double eps = static_cast<double>(eps_raw);
        if (std::isnan(eps) || std::isinf(eps) || eps <= 0) {
            eps = 1e-5;
        }
        eps = std::abs(eps);
        if (eps > 1.0) {
            eps = 1e-5;
        }
        
        auto result = torch::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps);
        
        if (offset < Size) {
            auto input2 = fuzzer_utils::createTensor(Data, Size, offset);
            if (input2.dim() >= 2 && input2.size(1) == num_features) {
                auto result2 = torch::batch_norm(input2, weight, bias, running_mean, running_var, training, momentum, eps);
            }
        }
        
        if (offset < Size) {
            try {
                auto weight_alt = torch::randn({num_features}, input.options());
                auto bias_alt = torch::randn({num_features}, input.options());
                auto result3 = torch::batch_norm(input, weight_alt, bias_alt, running_mean, running_var, training, momentum, eps);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            try {
                auto running_mean_alt = torch::randn({num_features}, input.options());
                auto running_var_alt = torch::abs(torch::randn({num_features}, input.options())) + eps;
                auto result4 = torch::batch_norm(input, weight, bias, running_mean_alt, running_var_alt, training, momentum, eps);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            try {
                auto result5 = torch::batch_norm(input, torch::Tensor(), torch::Tensor(), running_mean, running_var, training, momentum, eps);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            try {
                auto result6 = torch::batch_norm(input, weight, bias, torch::Tensor(), torch::Tensor(), training, momentum, eps);
            } catch (...) {
            }
        }
        
        if (offset < Size && input.dim() >= 3) {
            try {
                auto input_3d = input.view({input.size(0), input.size(1), -1});
                auto result7 = torch::batch_norm(input_3d, weight, bias, running_mean, running_var, training, momentum, eps);
            } catch (...) {
            }
        }
        
        if (offset < Size && input.dim() >= 4) {
            try {
                auto input_4d = input.view({input.size(0), input.size(1), input.size(2), -1});
                auto result8 = torch::batch_norm(input_4d, weight, bias, running_mean, running_var, training, momentum, eps);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            try {
                double extreme_eps = 1e-20;
                auto result9 = torch::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, extreme_eps);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            try {
                double extreme_momentum = 0.999999;
                auto result10 = torch::batch_norm(input, weight, bias, running_mean, running_var, training, extreme_momentum, eps);
            } catch (...) {
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