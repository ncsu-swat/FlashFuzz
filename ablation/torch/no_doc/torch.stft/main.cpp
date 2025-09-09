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
        
        if (offset >= Size) {
            return 0;
        }
        
        int64_t n_fft = static_cast<int64_t>(Data[offset++] % 64) + 1;
        
        if (offset >= Size) {
            return 0;
        }
        
        int64_t hop_length_raw = static_cast<int64_t>(Data[offset++] % 32) + 1;
        int64_t hop_length = std::min(hop_length_raw, n_fft);
        
        if (offset >= Size) {
            return 0;
        }
        
        int64_t win_length_raw = static_cast<int64_t>(Data[offset++] % 64) + 1;
        int64_t win_length = std::min(win_length_raw, n_fft);
        
        if (offset >= Size) {
            return 0;
        }
        
        bool center = (Data[offset++] % 2) == 1;
        bool normalized = (Data[offset++] % 2) == 1;
        bool onesided = (Data[offset++] % 2) == 1;
        bool return_complex = (Data[offset++] % 2) == 1;
        
        torch::Tensor window;
        if (offset < Size && (Data[offset++] % 2) == 1) {
            window = torch::hann_window(win_length, torch::TensorOptions().dtype(torch::kFloat));
        }
        
        if (input_tensor.dim() == 0) {
            input_tensor = input_tensor.unsqueeze(0);
        }
        
        if (input_tensor.dtype() != torch::kFloat && input_tensor.dtype() != torch::kDouble && 
            input_tensor.dtype() != torch::kComplexFloat && input_tensor.dtype() != torch::kComplexDouble) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        torch::Tensor result;
        if (window.defined()) {
            result = torch::stft(input_tensor, n_fft, hop_length, win_length, window, center, "reflect", normalized, onesided, return_complex);
        } else {
            result = torch::stft(input_tensor, n_fft, hop_length, win_length, {}, center, "reflect", normalized, onesided, return_complex);
        }
        
        if (offset < Size) {
            int64_t extreme_n_fft = static_cast<int64_t>(Data[offset++]) * 1000 + 1;
            torch::stft(input_tensor, extreme_n_fft, hop_length, win_length, {}, center, "reflect", normalized, onesided, return_complex);
        }
        
        if (offset < Size) {
            int64_t zero_hop = 0;
            torch::stft(input_tensor, n_fft, zero_hop, win_length, {}, center, "reflect", normalized, onesided, return_complex);
        }
        
        if (offset < Size) {
            int64_t negative_win = -1;
            torch::stft(input_tensor, n_fft, hop_length, negative_win, {}, center, "reflect", normalized, onesided, return_complex);
        }
        
        if (offset < Size && input_tensor.numel() > 0) {
            auto empty_tensor = torch::empty({0}, input_tensor.options());
            torch::stft(empty_tensor, n_fft, hop_length, win_length, {}, center, "reflect", normalized, onesided, return_complex);
        }
        
        if (offset < Size) {
            std::vector<std::string> pad_modes = {"reflect", "constant", "replicate"};
            std::string pad_mode = pad_modes[Data[offset++] % pad_modes.size()];
            torch::stft(input_tensor, n_fft, hop_length, win_length, {}, center, pad_mode, normalized, onesided, return_complex);
        }
        
        if (offset < Size && input_tensor.dim() >= 2) {
            auto squeezed = input_tensor.squeeze();
            torch::stft(squeezed, n_fft, hop_length, win_length, {}, center, "reflect", normalized, onesided, return_complex);
        }
        
        if (offset < Size) {
            auto mismatched_window = torch::hann_window(win_length + 10, torch::TensorOptions().dtype(torch::kFloat));
            torch::stft(input_tensor, n_fft, hop_length, win_length, mismatched_window, center, "reflect", normalized, onesided, return_complex);
        }
        
        if (offset < Size && input_tensor.numel() > 1000) {
            auto large_tensor = torch::randn({100000}, input_tensor.options());
            torch::stft(large_tensor, n_fft, hop_length, win_length, {}, center, "reflect", normalized, onesided, return_complex);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}