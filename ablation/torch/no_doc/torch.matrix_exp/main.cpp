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

        if (!input_tensor.is_floating_point() && !input_tensor.is_complex()) {
            input_tensor = input_tensor.to(torch::kFloat);
        }

        torch::Tensor result = torch::matrix_exp(input_tensor);

        if (offset < Size) {
            uint8_t scale_byte = Data[offset % Size];
            double scale = static_cast<double>(scale_byte) / 255.0 * 20.0 - 10.0;
            auto scaled_input = input_tensor * scale;
            torch::Tensor scaled_result = torch::matrix_exp(scaled_input);
        }

        if (offset + 1 < Size) {
            uint8_t noise_byte = Data[(offset + 1) % Size];
            double noise_level = static_cast<double>(noise_byte) / 255.0 * 0.1;
            auto noise = torch::randn_like(input_tensor) * noise_level;
            auto noisy_input = input_tensor + noise;
            torch::Tensor noisy_result = torch::matrix_exp(noisy_input);
        }

        auto zero_tensor = torch::zeros_like(input_tensor);
        torch::Tensor zero_result = torch::matrix_exp(zero_tensor);

        auto identity_shape = input_tensor.sizes().vec();
        auto identity_tensor = torch::eye(identity_shape[identity_shape.size()-1], 
                                        torch::TensorOptions().dtype(input_tensor.dtype()));
        if (input_tensor.dim() > 2) {
            std::vector<int64_t> batch_shape(identity_shape.begin(), identity_shape.end()-2);
            for (int64_t batch_size : batch_shape) {
                identity_tensor = identity_tensor.unsqueeze(0).expand({batch_size, -1, -1});
            }
        }
        torch::Tensor identity_result = torch::matrix_exp(identity_tensor);

        if (input_tensor.is_floating_point()) {
            auto inf_tensor = input_tensor.clone();
            inf_tensor.fill_(std::numeric_limits<float>::infinity());
            torch::Tensor inf_result = torch::matrix_exp(inf_tensor);

            auto neg_inf_tensor = input_tensor.clone();
            neg_inf_tensor.fill_(-std::numeric_limits<float>::infinity());
            torch::Tensor neg_inf_result = torch::matrix_exp(neg_inf_tensor);

            auto nan_tensor = input_tensor.clone();
            nan_tensor.fill_(std::numeric_limits<float>::quiet_NaN());
            torch::Tensor nan_result = torch::matrix_exp(nan_tensor);
        }

        if (offset + 2 < Size) {
            uint8_t batch_byte = Data[(offset + 2) % Size];
            if (batch_byte % 4 == 0 && input_tensor.dim() == 2) {
                auto batched_input = input_tensor.unsqueeze(0).expand({3, -1, -1});
                torch::Tensor batched_result = torch::matrix_exp(batched_input);
            }
        }

        auto large_tensor = torch::ones({2, 2}, input_tensor.options()) * 100.0;
        torch::Tensor large_result = torch::matrix_exp(large_tensor);

        auto small_tensor = torch::ones({2, 2}, input_tensor.options()) * 1e-10;
        torch::Tensor small_result = torch::matrix_exp(small_tensor);

        if (input_tensor.is_complex()) {
            auto real_part = torch::real(input_tensor);
            auto imag_part = torch::imag(input_tensor);
            auto modified_complex = torch::complex(imag_part, real_part);
            torch::Tensor complex_result = torch::matrix_exp(modified_complex);
        }

        auto transposed_input = input_tensor.transpose(-2, -1);
        torch::Tensor transposed_result = torch::matrix_exp(transposed_input);

        if (input_tensor.sizes()[-1] > 1) {
            auto singular_tensor = input_tensor.clone();
            singular_tensor.select(-1, 0).copy_(singular_tensor.select(-1, 1));
            torch::Tensor singular_result = torch::matrix_exp(singular_tensor);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}