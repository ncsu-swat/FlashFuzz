#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }

        torch::Tensor real_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor imag_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (real_tensor.dtype() != torch::kFloat && 
            real_tensor.dtype() != torch::kDouble && 
            real_tensor.dtype() != torch::kHalf) {
            auto options = torch::TensorOptions().dtype(torch::kFloat);
            real_tensor = real_tensor.to(torch::kFloat);
        }
        
        if (imag_tensor.dtype() != real_tensor.dtype()) {
            imag_tensor = imag_tensor.to(real_tensor.dtype());
        }
        
        if (real_tensor.sizes() != imag_tensor.sizes()) {
            auto min_numel = std::min(real_tensor.numel(), imag_tensor.numel());
            if (min_numel > 0) {
                real_tensor = real_tensor.flatten().slice(0, 0, min_numel);
                imag_tensor = imag_tensor.flatten().slice(0, 0, min_numel);
            } else {
                real_tensor = torch::zeros({1}, real_tensor.options());
                imag_tensor = torch::zeros({1}, imag_tensor.options());
            }
        }
        
        torch::Tensor complex_result = torch::complex(real_tensor, imag_tensor);
        
        if (offset < Size) {
            torch::Tensor out_tensor;
            try {
                if (real_tensor.dtype() == torch::kFloat) {
                    out_tensor = torch::empty(real_tensor.sizes(), torch::TensorOptions().dtype(torch::kComplexFloat));
                } else if (real_tensor.dtype() == torch::kDouble) {
                    out_tensor = torch::empty(real_tensor.sizes(), torch::TensorOptions().dtype(torch::kComplexDouble));
                } else {
                    out_tensor = torch::empty(real_tensor.sizes(), torch::TensorOptions().dtype(torch::kComplexFloat));
                }
                torch::complex_out(out_tensor, real_tensor, imag_tensor);
            } catch (...) {
            }
        }
        
        auto real_part = torch::real(complex_result);
        auto imag_part = torch::imag(complex_result);
        
        if (offset < Size) {
            try {
                auto zero_real = torch::zeros_like(real_tensor);
                auto zero_imag = torch::zeros_like(imag_tensor);
                torch::complex(zero_real, zero_imag);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            try {
                auto ones_real = torch::ones_like(real_tensor);
                auto ones_imag = torch::ones_like(imag_tensor);
                torch::complex(ones_real, ones_imag);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            try {
                auto inf_real = torch::full_like(real_tensor, std::numeric_limits<float>::infinity());
                auto inf_imag = torch::full_like(imag_tensor, std::numeric_limits<float>::infinity());
                torch::complex(inf_real, inf_imag);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            try {
                auto nan_real = torch::full_like(real_tensor, std::numeric_limits<float>::quiet_NaN());
                auto nan_imag = torch::full_like(imag_tensor, std::numeric_limits<float>::quiet_NaN());
                torch::complex(nan_real, nan_imag);
            } catch (...) {
            }
        }
        
        if (offset < Size && real_tensor.numel() > 0) {
            try {
                auto empty_real = torch::empty({0}, real_tensor.options());
                auto empty_imag = torch::empty({0}, imag_tensor.options());
                torch::complex(empty_real, empty_imag);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            try {
                auto large_real = torch::full_like(real_tensor, 1e38f);
                auto large_imag = torch::full_like(imag_tensor, 1e38f);
                torch::complex(large_real, large_imag);
            } catch (...) {
            }
        }
        
        if (offset < Size) {
            try {
                auto small_real = torch::full_like(real_tensor, 1e-38f);
                auto small_imag = torch::full_like(imag_tensor, 1e-38f);
                torch::complex(small_real, small_imag);
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