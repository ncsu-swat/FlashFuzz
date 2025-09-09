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
        
        auto real_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        auto imag_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (real_tensor.dtype() == torch::kComplexFloat || real_tensor.dtype() == torch::kComplexDouble ||
            imag_tensor.dtype() == torch::kComplexFloat || imag_tensor.dtype() == torch::kComplexDouble) {
            return 0;
        }
        
        torch::complex(real_tensor, imag_tensor);
        
        if (real_tensor.numel() == 0 || imag_tensor.numel() == 0) {
            torch::complex(real_tensor, imag_tensor);
        }
        
        if (real_tensor.sizes() != imag_tensor.sizes()) {
            torch::complex(real_tensor, imag_tensor);
        }
        
        auto scalar_real = torch::scalar_tensor(3.14, torch::kFloat);
        auto scalar_imag = torch::scalar_tensor(2.71, torch::kFloat);
        torch::complex(scalar_real, scalar_imag);
        
        if (real_tensor.numel() > 0) {
            torch::complex(real_tensor, scalar_imag);
            torch::complex(scalar_real, imag_tensor);
        }
        
        auto zero_real = torch::zeros_like(real_tensor);
        auto zero_imag = torch::zeros_like(imag_tensor);
        torch::complex(zero_real, zero_imag);
        
        auto inf_real = torch::full_like(real_tensor, std::numeric_limits<float>::infinity());
        auto inf_imag = torch::full_like(imag_tensor, std::numeric_limits<float>::infinity());
        torch::complex(inf_real, inf_imag);
        
        auto nan_real = torch::full_like(real_tensor, std::numeric_limits<float>::quiet_NaN());
        auto nan_imag = torch::full_like(imag_tensor, std::numeric_limits<float>::quiet_NaN());
        torch::complex(nan_real, nan_imag);
        
        if (real_tensor.dtype() != imag_tensor.dtype()) {
            torch::complex(real_tensor, imag_tensor);
        }
        
        auto large_real = torch::full_like(real_tensor, 1e38f);
        auto large_imag = torch::full_like(imag_tensor, 1e38f);
        torch::complex(large_real, large_imag);
        
        auto small_real = torch::full_like(real_tensor, 1e-38f);
        auto small_imag = torch::full_like(imag_tensor, 1e-38f);
        torch::complex(small_real, small_imag);
        
        if (real_tensor.dim() > 0 && imag_tensor.dim() > 0) {
            auto reshaped_real = real_tensor.view({-1});
            auto reshaped_imag = imag_tensor.view({-1});
            if (reshaped_real.numel() == reshaped_imag.numel()) {
                torch::complex(reshaped_real, reshaped_imag);
            }
        }
        
        auto neg_real = -real_tensor;
        auto neg_imag = -imag_tensor;
        torch::complex(neg_real, neg_imag);
        
        if (real_tensor.is_floating_point() && imag_tensor.is_floating_point()) {
            torch::complex(real_tensor, imag_tensor);
        }
        
        if (real_tensor.numel() == 1 && imag_tensor.numel() > 1) {
            torch::complex(real_tensor, imag_tensor);
        }
        
        if (real_tensor.numel() > 1 && imag_tensor.numel() == 1) {
            torch::complex(real_tensor, imag_tensor);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}