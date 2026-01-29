#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }
    
    try
    {
        // Need at least some data to proceed
        if (Size < 32) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters for stft from the data first
        int64_t n_fft = 64;  // Default reasonable value
        int64_t hop_length = 16;
        int64_t win_length = 64;
        bool normalized = false;
        bool onesided = true;
        bool return_complex = true;  // Default to true for modern API
        
        // Parse parameters if we have enough data
        if (offset + 24 <= Size) {
            // Extract n_fft (positive value, must be > 0)
            int64_t raw_n_fft;
            std::memcpy(&raw_n_fft, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            n_fft = (std::abs(raw_n_fft) % 512) + 4; // Ensure positive and reasonable size (4-515)
            
            // Extract hop_length (positive value, typically < n_fft)
            int64_t raw_hop_length;
            std::memcpy(&raw_hop_length, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            hop_length = (std::abs(raw_hop_length) % n_fft) + 1; // Ensure 1 <= hop_length <= n_fft
            
            // Extract win_length (must be <= n_fft)
            int64_t raw_win_length;
            std::memcpy(&raw_win_length, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            win_length = (std::abs(raw_win_length) % n_fft) + 1; // Ensure 1 <= win_length <= n_fft
        }
        
        // Extract boolean parameters
        if (offset + 3 <= Size) {
            normalized = Data[offset++] & 1;
            onesided = Data[offset++] & 1;
            return_complex = Data[offset++] & 1;
        }
        
        // Create input tensor - STFT requires 1D or 2D float tensor
        // Input length must be >= n_fft
        int64_t input_length = n_fft + (std::abs(static_cast<int64_t>(Data[offset % Size])) % 256);
        offset++;
        
        // Determine if we use 1D or 2D input
        bool use_2d = (offset < Size) && (Data[offset++] & 1);
        
        torch::Tensor input;
        if (use_2d) {
            int64_t batch_size = (offset < Size) ? ((Data[offset++] % 4) + 1) : 2;
            input = torch::randn({batch_size, input_length}, torch::kFloat32);
        } else {
            input = torch::randn({input_length}, torch::kFloat32);
        }
        
        // Seed from fuzzer data for reproducibility
        if (offset < Size) {
            unsigned int seed = 0;
            std::memcpy(&seed, Data + offset, std::min(sizeof(seed), Size - offset));
            torch::manual_seed(seed);
        }
        
        // Create window tensor
        torch::Tensor window;
        uint8_t window_type = (offset < Size) ? Data[offset++] : 0;
        
        // Choose window type based on the data
        switch (window_type % 5) {
            case 0:
                // Hann window
                window = torch::hann_window(win_length);
                break;
            case 1:
                // Hamming window
                window = torch::hamming_window(win_length);
                break;
            case 2:
                // Blackman window
                window = torch::blackman_window(win_length);
                break;
            case 3:
                // Bartlett window
                window = torch::bartlett_window(win_length);
                break;
            default:
                // No window (empty tensor)
                window = torch::Tensor();
                break;
        }
        
        // Apply stft operation
        torch::Tensor output = torch::stft(
            input,
            n_fft,
            hop_length,
            win_length,
            window,
            normalized,
            onesided,
            return_complex
        );
        
        // Ensure the output is not optimized away
        volatile bool output_exists = output.defined();
        (void)output_exists;
        
        // Additional operations to improve coverage
        if (output.defined()) {
            // Test accessing output dimensions
            volatile auto num_dims = output.dim();
            (void)num_dims;
            
            // Compute magnitude if complex
            if (output.is_complex()) {
                torch::Tensor magnitude = torch::abs(output);
                volatile auto mag_sum = magnitude.sum().item<float>();
                (void)mag_sum;
            }
        }
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
}