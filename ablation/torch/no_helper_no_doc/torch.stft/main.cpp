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
        if (Size < 32) return 0;

        // Extract input tensor dimensions and properties
        auto input_dims = extract_int_in_range(Data, Size, offset, 1, 4); // 1D to 4D tensor
        auto batch_size = extract_int_in_range(Data, Size, offset, 1, 8);
        auto signal_length = extract_int_in_range(Data, Size, offset, 16, 1024);
        
        // Create input tensor - can be 1D signal or batched
        torch::Tensor input;
        if (input_dims == 1) {
            input = torch::randn({signal_length}, torch::dtype(torch::kFloat32));
        } else if (input_dims == 2) {
            input = torch::randn({batch_size, signal_length}, torch::dtype(torch::kFloat32));
        } else if (input_dims == 3) {
            auto channels = extract_int_in_range(Data, Size, offset, 1, 4);
            input = torch::randn({batch_size, channels, signal_length}, torch::dtype(torch::kFloat32));
        } else {
            auto dim1 = extract_int_in_range(Data, Size, offset, 1, 4);
            auto dim2 = extract_int_in_range(Data, Size, offset, 1, 4);
            input = torch::randn({batch_size, dim1, dim2, signal_length}, torch::dtype(torch::kFloat32));
        }

        // Test with complex input sometimes
        auto use_complex = extract_bool(Data, Size, offset);
        if (use_complex) {
            input = torch::complex(input, torch::randn_like(input));
        }

        // Extract STFT parameters
        auto n_fft = extract_int_in_range(Data, Size, offset, 4, 512);
        
        // Ensure n_fft is power of 2 for better FFT performance (optional but common)
        n_fft = 1 << (int)std::log2(n_fft);
        
        auto hop_length_specified = extract_bool(Data, Size, offset);
        c10::optional<int64_t> hop_length = c10::nullopt;
        if (hop_length_specified) {
            hop_length = extract_int_in_range(Data, Size, offset, 1, n_fft);
        }

        auto win_length_specified = extract_bool(Data, Size, offset);
        c10::optional<int64_t> win_length = c10::nullopt;
        if (win_length_specified) {
            win_length = extract_int_in_range(Data, Size, offset, 1, n_fft);
        }

        // Window tensor - can be None or specified
        auto use_window = extract_bool(Data, Size, offset);
        c10::optional<torch::Tensor> window = c10::nullopt;
        if (use_window) {
            auto window_size = win_length.has_value() ? win_length.value() : n_fft;
            auto window_type = extract_int_in_range(Data, Size, offset, 0, 3);
            switch (window_type) {
                case 0:
                    window = torch::hann_window(window_size);
                    break;
                case 1:
                    window = torch::hamming_window(window_size);
                    break;
                case 2:
                    window = torch::blackman_window(window_size);
                    break;
                default:
                    window = torch::ones(window_size);
                    break;
            }
        }

        // Boolean parameters
        auto normalized = extract_bool(Data, Size, offset);
        auto onesided = extract_bool(Data, Size, offset);
        auto return_complex = extract_bool(Data, Size, offset);

        // Center parameter
        auto center = extract_bool(Data, Size, offset);

        // Pad mode
        auto pad_mode_idx = extract_int_in_range(Data, Size, offset, 0, 3);
        std::string pad_mode;
        switch (pad_mode_idx) {
            case 0: pad_mode = "reflect"; break;
            case 1: pad_mode = "constant"; break;
            case 2: pad_mode = "replicate"; break;
            default: pad_mode = "reflect"; break;
        }

        // Test basic STFT call
        auto result1 = torch::stft(input, n_fft);

        // Test with hop_length
        if (hop_length.has_value()) {
            auto result2 = torch::stft(input, n_fft, hop_length);
        }

        // Test with win_length
        if (win_length.has_value()) {
            auto result3 = torch::stft(input, n_fft, hop_length, win_length);
        }

        // Test with window
        if (window.has_value()) {
            auto result4 = torch::stft(input, n_fft, hop_length, win_length, window);
        }

        // Test with all parameters
        auto result5 = torch::stft(input, n_fft, hop_length, win_length, window, 
                                  normalized, onesided, return_complex);

        // Test edge cases
        
        // Very small n_fft
        if (signal_length >= 2) {
            auto small_result = torch::stft(input, 2);
        }

        // Large n_fft (but not larger than signal)
        if (signal_length >= 64) {
            auto large_n_fft = std::min(64, (int)signal_length);
            auto large_result = torch::stft(input, large_n_fft);
        }

        // Test with different hop lengths
        auto small_hop = std::max(1, n_fft / 4);
        auto large_hop = n_fft;
        
        auto hop_result1 = torch::stft(input, n_fft, small_hop);
        auto hop_result2 = torch::stft(input, n_fft, large_hop);

        // Test onesided vs twosided
        auto onesided_result = torch::stft(input, n_fft, hop_length, win_length, 
                                          window, normalized, true, return_complex);
        auto twosided_result = torch::stft(input, n_fft, hop_length, win_length, 
                                          window, normalized, false, return_complex);

        // Test normalized vs non-normalized
        auto norm_result = torch::stft(input, n_fft, hop_length, win_length, 
                                      window, true, onesided, return_complex);
        auto non_norm_result = torch::stft(input, n_fft, hop_length, win_length, 
                                          window, false, onesided, return_complex);

        // Test return_complex variations
        auto complex_result = torch::stft(input, n_fft, hop_length, win_length, 
                                         window, normalized, onesided, true);
        auto real_result = torch::stft(input, n_fft, hop_length, win_length, 
                                      window, normalized, onesided, false);

        // Test different input dtypes
        auto double_input = input.to(torch::kFloat64);
        auto double_result = torch::stft(double_input, n_fft);

        // Test with different devices if CUDA is available
        if (torch::cuda::is_available()) {
            auto cuda_input = input.to(torch::kCUDA);
            auto cuda_result = torch::stft(cuda_input, n_fft);
        }

        // Verify output shapes are reasonable
        if (result1.dim() >= 2) {
            auto shape = result1.sizes();
            // Basic sanity check - STFT should produce reasonable output dimensions
            if (shape[0] <= 0 || shape[1] <= 0) {
                throw std::runtime_error("Invalid STFT output shape");
            }
        }

        // Test with zero-length input (edge case)
        if (extract_bool(Data, Size, offset)) {
            auto zero_input = torch::zeros({0}, input.dtype());
            try {
                auto zero_result = torch::stft(zero_input, n_fft);
            } catch (const std::exception&) {
                // Expected to potentially fail
            }
        }

        // Test with very long signal
        if (extract_bool(Data, Size, offset) && signal_length < 100) {
            auto long_input = torch::randn({signal_length * 10}, input.dtype());
            auto long_result = torch::stft(long_input, n_fft);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}