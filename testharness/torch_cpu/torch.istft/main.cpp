#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 10) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract n_fft (must be positive and even for onesided)
        int16_t n_fft_raw;
        std::memcpy(&n_fft_raw, Data + offset, sizeof(int16_t));
        offset += sizeof(int16_t);
        int64_t n_fft = (std::abs(n_fft_raw) % 256 + 2) & ~1; // Ensure even, min 2
        
        // Extract hop_length (must be positive)
        int16_t hop_length_raw;
        std::memcpy(&hop_length_raw, Data + offset, sizeof(int16_t));
        offset += sizeof(int16_t);
        int64_t hop_length = std::abs(hop_length_raw) % (n_fft / 2) + 1;
        
        // Extract win_length (must be positive and <= n_fft)
        int16_t win_length_raw;
        std::memcpy(&win_length_raw, Data + offset, sizeof(int16_t));
        offset += sizeof(int16_t);
        int64_t win_length = std::abs(win_length_raw) % n_fft + 1;
        
        // Extract time frames
        int16_t time_frames_raw;
        std::memcpy(&time_frames_raw, Data + offset, sizeof(int16_t));
        offset += sizeof(int16_t);
        int64_t time_frames = std::abs(time_frames_raw) % 64 + 1;
        
        // Boolean parameters
        bool normalized = false;
        bool onesided = true;
        bool return_complex = false;
        bool center = true;
        
        if (offset < Size) {
            normalized = (Data[offset] & 0x01) != 0;
            onesided = (Data[offset] & 0x02) != 0;
            return_complex = (Data[offset] & 0x04) != 0;
            center = (Data[offset] & 0x08) != 0;
            offset++;
        }
        
        // Calculate frequency bins based on onesided parameter
        int64_t freq_bins = onesided ? (n_fft / 2 + 1) : n_fft;
        
        // Create complex spectrogram tensor with proper shape: (freq_bins, time_frames)
        // ISTFT expects complex input
        torch::Tensor complex_spectrogram = torch::randn(
            {freq_bins, time_frames}, 
            torch::TensorOptions().dtype(torch::kComplexFloat)
        );
        
        // Use fuzzer data to perturb the tensor values
        if (offset < Size) {
            int64_t numel = complex_spectrogram.numel();
            int64_t perturb_count = std::min(static_cast<int64_t>(Size - offset), numel);
            auto accessor = complex_spectrogram.view(-1);
            for (int64_t i = 0; i < perturb_count; i++) {
                float real_part = static_cast<float>(Data[offset + i]) / 255.0f - 0.5f;
                float imag_part = (offset + i + 1 < Size) ? 
                    static_cast<float>(Data[offset + i + 1]) / 255.0f - 0.5f : 0.0f;
                accessor[i] = c10::complex<float>(real_part, imag_part);
            }
        }
        
        // Create window tensor (1D tensor of size win_length)
        torch::Tensor window = torch::hann_window(win_length);
        
        // Apply istft operation
        try {
            torch::Tensor result = torch::istft(
                complex_spectrogram,
                n_fft,
                hop_length,
                win_length,
                window,
                center,
                normalized,
                onesided,
                c10::nullopt,  // length parameter
                return_complex
            );
            
            // Ensure the result is used to prevent optimization
            if (result.defined() && result.numel() > 0) {
                volatile auto sum = result.abs().sum().item<float>();
                (void)sum;
            }
        } catch (const c10::Error&) {
            // Expected errors from invalid parameter combinations
        }
        
        // Test batch input: (batch, freq_bins, time_frames)
        if (offset + 2 < Size) {
            int64_t batch_size = (Data[offset] % 4) + 1;
            torch::Tensor batch_spectrogram = torch::randn(
                {batch_size, freq_bins, time_frames},
                torch::TensorOptions().dtype(torch::kComplexFloat)
            );
            
            try {
                torch::Tensor batch_result = torch::istft(
                    batch_spectrogram,
                    n_fft,
                    hop_length,
                    win_length,
                    window,
                    center,
                    normalized,
                    onesided,
                    c10::nullopt,
                    return_complex
                );
                
                if (batch_result.defined() && batch_result.numel() > 0) {
                    volatile auto sum = batch_result.abs().sum().item<float>();
                    (void)sum;
                }
            } catch (const c10::Error&) {
                // Expected errors from invalid parameter combinations
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}