#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic parameters
        if (Size < 20) return 0;

        // Extract basic parameters
        int batch_size = extractInt(Data, Size, offset, 1, 8);
        int signal_length = extractInt(Data, Size, offset, 16, 2048);
        int n_fft = extractInt(Data, Size, offset, 8, 512);
        
        // Ensure n_fft is power of 2 for better testing
        n_fft = 1 << (int(std::log2(n_fft)) + 1);
        if (n_fft > 512) n_fft = 512;
        if (n_fft < 8) n_fft = 8;

        // Extract optional parameters
        bool use_hop_length = extractBool(Data, Size, offset);
        int hop_length = use_hop_length ? extractInt(Data, Size, offset, 1, n_fft) : n_fft / 4;
        
        bool use_win_length = extractBool(Data, Size, offset);
        int win_length = use_win_length ? extractInt(Data, Size, offset, 1, n_fft) : n_fft;
        
        bool use_window = extractBool(Data, Size, offset);
        bool center = extractBool(Data, Size, offset);
        bool normalized = extractBool(Data, Size, offset);
        bool onesided = extractBool(Data, Size, offset);
        bool return_complex = extractBool(Data, Size, offset);
        bool use_complex_input = extractBool(Data, Size, offset);
        
        // Extract pad_mode
        int pad_mode_idx = extractInt(Data, Size, offset, 0, 4);
        std::vector<std::string> pad_modes = {"reflect", "constant", "replicate", "circular"};
        std::string pad_mode = pad_modes[pad_mode_idx];

        // Create input tensor
        torch::Tensor input;
        if (use_complex_input) {
            // Complex input
            auto real_part = torch::randn({batch_size, signal_length}, torch::kFloat32);
            auto imag_part = torch::randn({batch_size, signal_length}, torch::kFloat32);
            input = torch::complex(real_part, imag_part);
            // For complex input, onesided must be False
            onesided = false;
        } else {
            // Real input
            input = torch::randn({batch_size, signal_length}, torch::kFloat32);
        }

        // Test 1D input as well
        bool use_1d = extractBool(Data, Size, offset);
        if (use_1d && batch_size == 1) {
            input = input.squeeze(0); // Remove batch dimension
        }

        // Create window tensor if needed
        torch::Tensor window;
        if (use_window) {
            // Test different window types
            int window_type = extractInt(Data, Size, offset, 0, 3);
            switch (window_type) {
                case 0:
                    window = torch::hann_window(win_length);
                    break;
                case 1:
                    window = torch::hamming_window(win_length);
                    break;
                case 2:
                    window = torch::blackman_window(win_length);
                    break;
                default:
                    window = torch::ones(win_length);
                    break;
            }
            
            // Sometimes make window complex to test edge cases
            if (use_complex_input && extractBool(Data, Size, offset)) {
                auto imag_window = torch::zeros_like(window);
                window = torch::complex(window, imag_window);
            }
        }

        // Test edge cases
        if (extractBool(Data, Size, offset)) {
            // Test very small signal
            if (input.dim() == 1) {
                input = torch::randn({n_fft / 2});
            } else {
                input = torch::randn({batch_size, n_fft / 2});
            }
        }

        if (extractBool(Data, Size, offset)) {
            // Test signal shorter than n_fft
            int short_length = extractInt(Data, Size, offset, 1, n_fft - 1);
            if (input.dim() == 1) {
                input = torch::randn({short_length});
            } else {
                input = torch::randn({batch_size, short_length});
            }
        }

        // Call torch::stft with different parameter combinations
        torch::Tensor result;
        
        // Test basic call
        if (use_window) {
            result = torch::stft(input, n_fft, hop_length, win_length, window, 
                               center, pad_mode, normalized, onesided, return_complex);
        } else {
            result = torch::stft(input, n_fft, hop_length, win_length, {}, 
                               center, pad_mode, normalized, onesided, return_complex);
        }

        // Verify output shape
        auto sizes = result.sizes();
        if (return_complex) {
            // Should be (B?, N, T) for complex output
            if (input.dim() == 1) {
                assert(result.dim() == 2); // (N, T)
            } else {
                assert(result.dim() == 3); // (B, N, T)
            }
        } else {
            // Should be (B?, N, T, 2) for real output
            if (input.dim() == 1) {
                assert(result.dim() == 3); // (N, T, 2)
                assert(result.size(-1) == 2);
            } else {
                assert(result.dim() == 4); // (B, N, T, 2)
                assert(result.size(-1) == 2);
            }
        }

        // Test with default parameters
        if (extractBool(Data, Size, offset)) {
            auto result2 = torch::stft(input, n_fft, {}, {}, {}, true, "reflect", 
                                     false, onesided, return_complex);
        }

        // Test with minimal parameters
        if (extractBool(Data, Size, offset)) {
            auto result3 = torch::stft(input, n_fft, {}, {}, {}, {}, {}, {}, 
                                     onesided, return_complex);
        }

        // Test different dtypes
        if (extractBool(Data, Size, offset)) {
            auto input_double = input.to(torch::kFloat64);
            auto result_double = torch::stft(input_double, n_fft, hop_length, win_length, 
                                           use_window ? window.to(torch::kFloat64) : torch::Tensor{}, 
                                           center, pad_mode, normalized, onesided, return_complex);
        }

        // Test edge case: win_length > n_fft (should be clamped)
        if (extractBool(Data, Size, offset)) {
            int large_win_length = n_fft + extractInt(Data, Size, offset, 1, 64);
            auto large_window = torch::hann_window(large_win_length);
            auto result4 = torch::stft(input, n_fft, hop_length, large_win_length, 
                                     large_window, center, pad_mode, normalized, 
                                     onesided, return_complex);
        }

        // Test with zero hop_length (edge case)
        if (extractBool(Data, Size, offset)) {
            try {
                auto result5 = torch::stft(input, n_fft, 0, win_length, 
                                         use_window ? window : torch::Tensor{}, 
                                         center, pad_mode, normalized, onesided, return_complex);
            } catch (...) {
                // Expected to fail
            }
        }

        // Test very large hop_length
        if (extractBool(Data, Size, offset)) {
            int large_hop = signal_length + 100;
            auto result6 = torch::stft(input, n_fft, large_hop, win_length, 
                                     use_window ? window : torch::Tensor{}, 
                                     center, pad_mode, normalized, onesided, return_complex);
        }

        // Test empty input
        if (extractBool(Data, Size, offset)) {
            torch::Tensor empty_input;
            if (input.dim() == 1) {
                empty_input = torch::empty({0});
            } else {
                empty_input = torch::empty({batch_size, 0});
            }
            try {
                auto result7 = torch::stft(empty_input, n_fft, hop_length, win_length, 
                                         use_window ? window : torch::Tensor{}, 
                                         center, pad_mode, normalized, onesided, return_complex);
            } catch (...) {
                // May fail, which is acceptable
            }
        }

        // Force some operations on the result to ensure it's valid
        auto result_sum = result.sum();
        auto result_mean = result.mean();
        
        // Test accessing elements
        if (result.numel() > 0) {
            auto first_elem = result.flatten()[0];
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}