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

        auto input = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (input.dim() < 2) {
            auto shape = input.sizes().vec();
            while (shape.size() < 2) {
                shape.push_back(1);
            }
            input = input.reshape(shape);
        }
        
        if (input.size(0) == 0 || input.size(1) == 0) {
            input = torch::ones({1, 1}, input.options());
        }
        
        if (!input.dtype().is_floating_point() && !input.dtype().is_complex()) {
            input = input.to(torch::kFloat);
        }
        
        auto result = torch::geqrf(input);
        auto a = std::get<0>(result);
        auto tau = std::get<1>(result);
        
        if (offset < Size) {
            auto out_a = torch::empty_like(a);
            auto out_tau = torch::empty_like(tau);
            auto out_tuple = std::make_tuple(out_a, out_tau);
            torch::geqrf_out(out_tuple, input);
        }
        
        if (offset + 1 < Size) {
            uint8_t shape_modifier = Data[offset];
            if (shape_modifier % 4 == 0 && input.size(0) > 1) {
                auto sliced = input.slice(0, 0, input.size(0) - 1);
                torch::geqrf(sliced);
            } else if (shape_modifier % 4 == 1 && input.size(1) > 1) {
                auto sliced = input.slice(1, 0, input.size(1) - 1);
                torch::geqrf(sliced);
            } else if (shape_modifier % 4 == 2) {
                auto transposed = input.transpose(0, 1);
                torch::geqrf(transposed);
            } else if (shape_modifier % 4 == 3) {
                auto contiguous = input.contiguous();
                torch::geqrf(contiguous);
            }
        }
        
        if (offset + 2 < Size) {
            uint8_t batch_flag = Data[offset + 1];
            if (batch_flag % 2 == 0 && input.dim() == 2) {
                auto batched = input.unsqueeze(0);
                torch::geqrf(batched);
            }
        }
        
        if (offset + 3 < Size) {
            uint8_t extreme_flag = Data[offset + 2];
            if (extreme_flag % 8 == 0) {
                auto large_input = torch::randn({100, 50}, input.options());
                torch::geqrf(large_input);
            } else if (extreme_flag % 8 == 1) {
                auto tall_input = torch::randn({50, 10}, input.options());
                torch::geqrf(tall_input);
            } else if (extreme_flag % 8 == 2) {
                auto wide_input = torch::randn({10, 50}, input.options());
                torch::geqrf(wide_input);
            } else if (extreme_flag % 8 == 3) {
                auto square_input = torch::randn({25, 25}, input.options());
                torch::geqrf(square_input);
            } else if (extreme_flag % 8 == 4) {
                auto tiny_input = torch::randn({2, 2}, input.options());
                torch::geqrf(tiny_input);
            } else if (extreme_flag % 8 == 5) {
                auto single_col = torch::randn({10, 1}, input.options());
                torch::geqrf(single_col);
            } else if (extreme_flag % 8 == 6) {
                auto single_row = torch::randn({1, 10}, input.options());
                torch::geqrf(single_row);
            } else if (extreme_flag % 8 == 7) {
                auto minimal = torch::randn({1, 1}, input.options());
                torch::geqrf(minimal);
            }
        }
        
        if (input.dtype().is_complex()) {
            auto real_part = torch::real(input);
            auto imag_part = torch::imag(input);
            if (real_part.numel() > 0 && imag_part.numel() > 0) {
                torch::geqrf(real_part);
                torch::geqrf(imag_part);
            }
        }
        
        if (offset + 4 < Size) {
            uint8_t special_flag = Data[offset + 3];
            if (special_flag % 16 == 0) {
                auto zeros_input = torch::zeros_like(input);
                torch::geqrf(zeros_input);
            } else if (special_flag % 16 == 1) {
                auto ones_input = torch::ones_like(input);
                torch::geqrf(ones_input);
            } else if (special_flag % 16 == 2) {
                auto eye_size = std::min(input.size(0), input.size(1));
                if (eye_size > 0) {
                    auto eye_input = torch::eye(eye_size, input.options());
                    if (input.size(0) != eye_size || input.size(1) != eye_size) {
                        eye_input = torch::nn::functional::pad(eye_input, 
                            torch::nn::functional::PadFuncOptions({0, input.size(1) - eye_size, 0, input.size(0) - eye_size}));
                    }
                    torch::geqrf(eye_input);
                }
            } else if (special_flag % 16 == 3) {
                auto inf_input = torch::full_like(input, std::numeric_limits<float>::infinity());
                torch::geqrf(inf_input);
            } else if (special_flag % 16 == 4) {
                auto ninf_input = torch::full_like(input, -std::numeric_limits<float>::infinity());
                torch::geqrf(ninf_input);
            } else if (special_flag % 16 == 5) {
                auto nan_input = torch::full_like(input, std::numeric_limits<float>::quiet_NaN());
                torch::geqrf(nan_input);
            } else if (special_flag % 16 == 6) {
                auto very_large = torch::full_like(input, 1e20f);
                torch::geqrf(very_large);
            } else if (special_flag % 16 == 7) {
                auto very_small = torch::full_like(input, 1e-20f);
                torch::geqrf(very_small);
            } else if (special_flag % 16 == 8) {
                auto negative_input = -torch::abs(input);
                torch::geqrf(negative_input);
            } else if (special_flag % 16 == 9) {
                auto positive_input = torch::abs(input);
                torch::geqrf(positive_input);
            } else if (special_flag % 16 == 10) {
                auto random_signs = torch::randint(0, 2, input.sizes(), input.options()) * 2 - 1;
                auto mixed_signs = input * random_signs.to(input.dtype());
                torch::geqrf(mixed_signs);
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