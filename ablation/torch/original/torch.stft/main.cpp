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
        
        if (input.dim() == 0) {
            input = input.unsqueeze(0);
        }
        if (input.dim() > 2) {
            input = input.flatten(1);
        }
        
        uint8_t n_fft_byte = Data[offset++];
        int n_fft = 1 + (n_fft_byte % 512);
        
        c10::optional<int64_t> hop_length = c10::nullopt;
        c10::optional<int64_t> win_length = c10::nullopt;
        c10::optional<torch::Tensor> window = c10::nullopt;
        bool center = true;
        std::string pad_mode = "reflect";
        bool normalized = false;
        c10::optional<bool> onesided = c10::nullopt;
        bool return_complex = true;
        
        if (offset < Size) {
            uint8_t hop_byte = Data[offset++];
            if (hop_byte % 4 == 0) {
                hop_length = 1 + (hop_byte % 256);
            }
        }
        
        if (offset < Size) {
            uint8_t win_byte = Data[offset++];
            if (win_byte % 4 == 0) {
                win_length = 1 + (win_byte % n_fft);
            }
        }
        
        if (offset < Size) {
            uint8_t window_byte = Data[offset++];
            if (window_byte % 3 == 0) {
                int win_size = win_length.has_value() ? win_length.value() : n_fft;
                if (window_byte % 6 == 0) {
                    window = torch::hann_window(win_size);
                } else {
                    window = torch::ones(win_size);
                }
            }
        }
        
        if (offset < Size) {
            uint8_t flags_byte = Data[offset++];
            center = (flags_byte & 1) != 0;
            normalized = (flags_byte & 2) != 0;
            return_complex = (flags_byte & 4) != 0;
            
            if (flags_byte & 8) {
                onesided = (flags_byte & 16) != 0;
            }
            
            int pad_mode_idx = (flags_byte >> 5) % 4;
            switch (pad_mode_idx) {
                case 0: pad_mode = "reflect"; break;
                case 1: pad_mode = "constant"; break;
                case 2: pad_mode = "replicate"; break;
                case 3: pad_mode = "circular"; break;
            }
        }
        
        if (input.is_complex() && onesided.has_value() && onesided.value()) {
            onesided = false;
        }
        
        torch::Tensor result = torch::stft(
            input,
            n_fft,
            hop_length,
            win_length,
            window,
            center,
            pad_mode,
            normalized,
            onesided,
            return_complex
        );
        
        if (result.numel() > 0) {
            auto sum = torch::sum(result);
            if (sum.is_complex()) {
                auto real_part = torch::real(sum);
                auto imag_part = torch::imag(sum);
                volatile float real_val = real_part.item<float>();
                volatile float imag_val = imag_part.item<float>();
            } else {
                volatile float val = sum.item<float>();
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