#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create complex spectrogram tensor
        torch::Tensor complex_spectrogram = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for istft
        int64_t n_fft = 400;
        int64_t hop_length = 100;
        int64_t win_length = 400;
        bool normalized = false;
        bool onesided = true;
        bool return_complex = false;
        
        // If we have more data, use it to set parameters
        if (offset + 6 <= Size) {
            // Extract n_fft (must be positive)
            int16_t n_fft_raw;
            std::memcpy(&n_fft_raw, Data + offset, sizeof(int16_t));
            offset += sizeof(int16_t);
            n_fft = std::abs(n_fft_raw) % 1024 + 1;
            
            // Extract hop_length (must be positive)
            int16_t hop_length_raw;
            std::memcpy(&hop_length_raw, Data + offset, sizeof(int16_t));
            offset += sizeof(int16_t);
            hop_length = std::abs(hop_length_raw) % 512 + 1;
            
            // Extract win_length (must be positive and <= n_fft)
            int16_t win_length_raw;
            std::memcpy(&win_length_raw, Data + offset, sizeof(int16_t));
            offset += sizeof(int16_t);
            win_length = std::abs(win_length_raw) % (n_fft + 1);
            if (win_length == 0) win_length = n_fft;
        }
        
        // If we have more data, use it for boolean parameters
        if (offset < Size) {
            normalized = (Data[offset] & 0x01) != 0;
            onesided = (Data[offset] & 0x02) != 0;
            return_complex = (Data[offset] & 0x04) != 0;
            offset++;
        }
        
        // Create window tensor if we have more data
        torch::Tensor window;
        if (offset < Size) {
            try {
                window = fuzzer_utils::createTensor(Data, Size, offset);
            } catch (const std::exception&) {
                // If window creation fails, use default window (None)
            }
        }
        
        // Apply istft operation
        torch::Tensor result;
        
        // Handle different cases for window parameter
        if (window.defined()) {
            result = torch::istft(
                complex_spectrogram,
                n_fft,
                torch::optional<int64_t>(hop_length),
                torch::optional<int64_t>(win_length),
                window,
                true, // center parameter
                normalized,
                torch::optional<bool>(onesided),
                torch::optional<int64_t>(), // length parameter (optional)
                return_complex
            );
        } else {
            result = torch::istft(
                complex_spectrogram,
                n_fft,
                torch::optional<int64_t>(hop_length),
                torch::optional<int64_t>(win_length),
                torch::optional<torch::Tensor>(), // No window
                true, // center parameter
                normalized,
                torch::optional<bool>(onesided),
                torch::optional<int64_t>(), // length parameter (optional)
                return_complex
            );
        }
        
        // Ensure the result is used to prevent optimization
        if (result.defined()) {
            volatile auto numel = result.numel();
            if (numel > 0) {
                volatile auto item = result.item<float>();
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}