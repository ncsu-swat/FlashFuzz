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
        if (offset >= Size) return 0;
        
        auto tau = fuzzer_utils::createTensor(Data, Size, offset);
        if (offset >= Size) return 0;
        
        auto other = fuzzer_utils::createTensor(Data, Size, offset);
        if (offset >= Size) return 0;

        if (input.numel() == 0 || tau.numel() == 0 || other.numel() == 0) {
            return 0;
        }

        if (input.dim() < 2 || tau.dim() < 1 || other.dim() < 2) {
            return 0;
        }

        bool left = (Data[offset % Size] & 1) == 1;
        offset++;
        bool transpose = (Data[offset % Size] & 1) == 1;
        offset++;

        auto input_float = input.to(torch::kFloat);
        auto tau_float = tau.to(torch::kFloat);
        auto other_float = other.to(torch::kFloat);

        int64_t batch_dims = std::max({input_float.dim() - 2, tau_float.dim() - 1, other_float.dim() - 2});
        
        std::vector<int64_t> batch_shape;
        for (int64_t i = 0; i < batch_dims; ++i) {
            int64_t batch_size = 1;
            if (i < input_float.dim() - 2) batch_size = std::max(batch_size, input_float.size(i));
            if (i < tau_float.dim() - 1) batch_size = std::max(batch_size, tau_float.size(i));
            if (i < other_float.dim() - 2) batch_size = std::max(batch_size, other_float.size(i));
            batch_shape.push_back(batch_size);
        }

        int64_t m = other_float.size(-2);
        int64_t n = other_float.size(-1);
        int64_t k = input_float.size(-1);
        int64_t mn = left ? m : n;

        std::vector<int64_t> input_shape = batch_shape;
        input_shape.push_back(mn);
        input_shape.push_back(k);
        
        std::vector<int64_t> tau_shape = batch_shape;
        tau_shape.push_back(std::min(mn, k));
        
        std::vector<int64_t> other_shape = batch_shape;
        other_shape.push_back(m);
        other_shape.push_back(n);

        input_float = input_float.view(input_shape);
        tau_float = tau_float.view(tau_shape);
        other_float = other_float.view(other_shape);

        auto result = torch::ormqr(input_float, tau_float, other_float, left, transpose);

        auto input_double = input_float.to(torch::kDouble);
        auto tau_double = tau_float.to(torch::kDouble);
        auto other_double = other_float.to(torch::kDouble);
        auto result_double = torch::ormqr(input_double, tau_double, other_double, left, transpose);

        auto input_cfloat = input_float.to(torch::kComplexFloat);
        auto tau_cfloat = tau_float.to(torch::kComplexFloat);
        auto other_cfloat = other_float.to(torch::kComplexFloat);
        auto result_cfloat = torch::ormqr(input_cfloat, tau_cfloat, other_cfloat, left, transpose);

        auto input_cdouble = input_float.to(torch::kComplexDouble);
        auto tau_cdouble = tau_float.to(torch::kComplexDouble);
        auto other_cdouble = other_float.to(torch::kComplexDouble);
        auto result_cdouble = torch::ormqr(input_cdouble, tau_cdouble, other_cdouble, left, transpose);

        torch::Tensor out_tensor = torch::empty_like(result);
        torch::ormqr_out(out_tensor, input_float, tau_float, other_float, left, transpose);

        auto zero_input = torch::zeros_like(input_float);
        auto zero_tau = torch::zeros_like(tau_float);
        auto zero_other = torch::zeros_like(other_float);
        auto zero_result = torch::ormqr(zero_input, zero_tau, zero_other, left, transpose);

        auto ones_input = torch::ones_like(input_float);
        auto ones_tau = torch::ones_like(tau_float);
        auto ones_other = torch::ones_like(other_float);
        auto ones_result = torch::ormqr(ones_input, ones_tau, ones_other, left, transpose);

        auto neg_input = -input_float;
        auto neg_tau = -tau_float;
        auto neg_other = -other_float;
        auto neg_result = torch::ormqr(neg_input, neg_tau, neg_other, left, transpose);

        auto large_input = input_float * 1e6;
        auto large_tau = tau_float * 1e6;
        auto large_other = other_float * 1e6;
        auto large_result = torch::ormqr(large_input, large_tau, large_other, left, transpose);

        auto small_input = input_float * 1e-6;
        auto small_tau = tau_float * 1e-6;
        auto small_other = other_float * 1e-6;
        auto small_result = torch::ormqr(small_input, small_tau, small_other, left, transpose);

        if (input_float.requires_grad()) {
            input_float.requires_grad_(true);
            tau_float.requires_grad_(true);
            other_float.requires_grad_(true);
            auto grad_result = torch::ormqr(input_float, tau_float, other_float, left, transpose);
            auto loss = grad_result.sum();
            loss.backward();
        }

        torch::ormqr(input_float, tau_float, other_float, !left, transpose);
        torch::ormqr(input_float, tau_float, other_float, left, !transpose);
        torch::ormqr(input_float, tau_float, other_float, !left, !transpose);

        if (input_float.dim() > 2) {
            auto squeezed_input = input_float.squeeze(0);
            auto squeezed_tau = tau_float.squeeze(0);
            auto squeezed_other = other_float.squeeze(0);
            if (squeezed_input.dim() >= 2 && squeezed_tau.dim() >= 1 && squeezed_other.dim() >= 2) {
                auto squeezed_result = torch::ormqr(squeezed_input, squeezed_tau, squeezed_other, left, transpose);
            }
        }

        auto unsqueezed_input = input_float.unsqueeze(0);
        auto unsqueezed_tau = tau_float.unsqueeze(0);
        auto unsqueezed_other = other_float.unsqueeze(0);
        auto unsqueezed_result = torch::ormqr(unsqueezed_input, unsqueezed_tau, unsqueezed_other, left, transpose);

        auto transposed_input = input_float.transpose(-2, -1);
        auto transposed_other = other_float.transpose(-2, -1);
        if (transposed_input.size(-2) == (left ? transposed_other.size(-1) : transposed_other.size(-2))) {
            auto transposed_result = torch::ormqr(transposed_input, tau_float, transposed_other, left, transpose);
        }

        auto contiguous_input = input_float.contiguous();
        auto contiguous_tau = tau_float.contiguous();
        auto contiguous_other = other_float.contiguous();
        auto contiguous_result = torch::ormqr(contiguous_input, contiguous_tau, contiguous_other, left, transpose);

        auto non_contiguous_input = input_float.transpose(-2, -1).transpose(-2, -1);
        auto non_contiguous_tau = tau_float.unsqueeze(-1).squeeze(-1);
        auto non_contiguous_other = other_float.transpose(-2, -1).transpose(-2, -1);
        auto non_contiguous_result = torch::ormqr(non_contiguous_input, non_contiguous_tau, non_contiguous_other, left, transpose);

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}