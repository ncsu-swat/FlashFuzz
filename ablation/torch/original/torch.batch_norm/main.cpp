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
            uint8_t cudnn_byte = Data[offset++];
            bool cudnn_enabled = (cudnn_byte % 2) == 1;
            auto result3 = torch::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
        }
        
        if (offset < Size && input.dim() >= 3) {
            try {
                auto result4 = torch::batch_norm(input, torch::Tensor(), torch::Tensor(), running_mean, running_var, training, momentum, eps);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            try {
                auto empty_running_mean = torch::empty({0}, input.options());
                auto empty_running_var = torch::empty({0}, input.options());
                auto result5 = torch::batch_norm(input, weight, bias, empty_running_mean, empty_running_var, training, momentum, eps);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            try {
                auto wrong_size_weight = torch::ones({num_features + 1}, input.options());
                auto result6 = torch::batch_norm(input, wrong_size_weight, bias, running_mean, running_var, training, momentum, eps);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            try {
                auto wrong_size_bias = torch::zeros({num_features - 1}, input.options());
                auto result7 = torch::batch_norm(input, weight, wrong_size_bias, running_mean, running_var, training, momentum, eps);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            try {
                auto wrong_size_running_mean = torch::zeros({num_features * 2}, input.options());
                auto result8 = torch::batch_norm(input, weight, bias, wrong_size_running_mean, running_var, training, momentum, eps);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            try {
                auto wrong_size_running_var = torch::ones({1}, input.options());
                auto result9 = torch::batch_norm(input, weight, bias, running_mean, wrong_size_running_var, training, momentum, eps);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            try {
                auto negative_running_var = torch::full({num_features}, -1.0, input.options());
                auto result10 = torch::batch_norm(input, weight, bias, running_mean, negative_running_var, training, momentum, eps);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            try {
                auto inf_weight = torch::full({num_features}, std::numeric_limits<float>::infinity(), input.options());
                auto result11 = torch::batch_norm(input, inf_weight, bias, running_mean, running_var, training, momentum, eps);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            try {
                auto nan_bias = torch::full({num_features}, std::numeric_limits<float>::quiet_NaN(), input.options());
                auto result12 = torch::batch_norm(input, weight, nan_bias, running_mean, running_var, training, momentum, eps);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            try {
                auto result13 = torch::batch_norm(input, weight, bias, running_mean, running_var, training, -momentum, eps);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            try {
                auto result14 = torch::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, -eps);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            try {
                auto result15 = torch::batch_norm(input, weight, bias, running_mean, running_var, training, 2.0, eps);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            try {
                auto result16 = torch::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, 0.0);
            } catch (...) {
            }
        }
        
        if (offset < Size && input.dim() == 4) {
            try {
                auto permuted_input = input.permute({0, 2, 3, 1});
                auto result17 = torch::batch_norm(permuted_input, weight, bias, running_mean, running_var, training, momentum, eps);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            try {
                auto squeezed_input = input.squeeze();
                if (squeezed_input.dim() >= 2) {
                    int64_t squeezed_features = squeezed_input.size(1);
                    if (squeezed_features == num_features) {
                        auto result18 = torch::batch_norm(squeezed_input, weight, bias, running_mean, running_var, training, momentum, eps);
                    }
                }
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            try {
                auto unsqueezed_input = input.unsqueeze(0);
                auto result19 = torch::batch_norm(unsqueezed_input, weight, bias, running_mean, running_var, training, momentum, eps);
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