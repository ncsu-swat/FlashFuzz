#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for stft from the remaining data
        int64_t n_fft = 400;
        int64_t hop_length = 100;
        int64_t win_length = 400;
        bool normalized = false;
        bool onesided = true;
        bool return_complex = false;
        
        // Parse parameters if we have more data
        if (offset + 6 <= Size) {
            // Extract n_fft (positive value)
            int64_t raw_n_fft;
            std::memcpy(&raw_n_fft, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            n_fft = std::abs(raw_n_fft) % 1024 + 1; // Ensure positive and reasonable size
            
            // Extract hop_length (positive value)
            int64_t raw_hop_length;
            std::memcpy(&raw_hop_length, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            hop_length = std::abs(raw_hop_length) % (n_fft + 1) + 1; // Ensure positive and reasonable size
            
            // Extract win_length (positive value, can be null)
            int64_t raw_win_length;
            std::memcpy(&raw_win_length, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            if (raw_win_length < 0) {
                win_length = n_fft; // Default to n_fft if negative
            } else {
                win_length = raw_win_length % 1024 + 1; // Ensure positive and reasonable size
            }
            
            // Extract boolean parameters if we have more data
            if (offset + 3 <= Size) {
                normalized = Data[offset++] & 1;
                onesided = Data[offset++] & 1;
                return_complex = Data[offset++] & 1;
            }
        }
        
        // Create window tensor
        torch::Tensor window;
        if (offset + 1 <= Size) {
            uint8_t window_type = Data[offset++];
            
            // Choose window type based on the data
            if (window_type % 4 == 0) {
                // Hann window
                window = torch::hann_window(win_length);
            } else if (window_type % 4 == 1) {
                // Hamming window
                window = torch::hamming_window(win_length);
            } else if (window_type % 4 == 2) {
                // Blackman window
                window = torch::blackman_window(win_length);
            } else {
                // Try with null window
                window = torch::Tensor();
            }
        } else {
            // Default to Hann window if no more data
            window = torch::hann_window(win_length);
        }
        
        // Apply stft operation
        torch::Tensor output = torch::stft(
            input,
            n_fft,
            torch::optional<int64_t>(hop_length),
            torch::optional<int64_t>(win_length),
            window,
            normalized,
            onesided,
            torch::optional<bool>(return_complex)
        );
        
        // Ensure the output is not optimized away
        volatile bool output_exists = output.defined();
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
