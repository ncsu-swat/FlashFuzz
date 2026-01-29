#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 4) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract window_length from data (1 to 1000)
        uint16_t raw_window_length;
        std::memcpy(&raw_window_length, Data + offset, sizeof(uint16_t));
        offset += sizeof(uint16_t);
        int64_t window_length = static_cast<int64_t>(raw_window_length % 1000) + 1;
        
        // Parse window function selector
        uint8_t window_fn_selector = 0;
        if (offset < Size) {
            window_fn_selector = Data[offset++] % 5;
        }
        
        // Parse periodic flag
        bool periodic = true;
        if (offset < Size) {
            periodic = (Data[offset++] % 2) == 0;
        }
        
        // Parse beta for kaiser window (ensure valid range 0-50)
        double beta = 12.0;
        if (offset < Size) {
            beta = static_cast<double>(Data[offset++]) / 5.0; // 0 to ~51
        }
        
        // Parse dtype selector
        uint8_t dtype_selector = 0;
        if (offset < Size) {
            dtype_selector = Data[offset++] % 3;
        }
        
        torch::TensorOptions options;
        switch (dtype_selector) {
            case 0:
                options = options.dtype(torch::kFloat32);
                break;
            case 1:
                options = options.dtype(torch::kFloat64);
                break;
            case 2:
                options = options.dtype(torch::kFloat16);
                break;
        }
        
        // Test the selected window function
        try {
            torch::Tensor result;
            
            switch (window_fn_selector) {
                case 0: {
                    // Hann window
                    result = torch::hann_window(window_length, periodic, options);
                    break;
                }
                case 1: {
                    // Hamming window
                    result = torch::hamming_window(window_length, periodic, options);
                    // Also test with alpha/beta parameters
                    if (offset + 2 <= Size) {
                        double alpha = 0.54 + (Data[offset++] % 46) / 100.0; // 0.54 to 1.0
                        double ham_beta = 1.0 - alpha;
                        result = torch::hamming_window(window_length, periodic, alpha, ham_beta, options);
                    }
                    break;
                }
                case 2: {
                    // Bartlett window
                    result = torch::bartlett_window(window_length, periodic, options);
                    break;
                }
                case 3: {
                    // Blackman window
                    result = torch::blackman_window(window_length, periodic, options);
                    break;
                }
                case 4: {
                    // Kaiser window
                    result = torch::kaiser_window(window_length, periodic, beta, options);
                    break;
                }
            }
            
            // Verify output properties
            if (result.defined()) {
                (void)result.size(0);
                (void)result.sum();
            }
            
        } catch (const c10::Error& e) {
            // Expected PyTorch errors (invalid parameters, etc.)
        }
        
        // Additional coverage: test all window functions with base parameters
        try {
            torch::Tensor h1 = torch::hann_window(window_length);
            torch::Tensor h2 = torch::hamming_window(window_length);
            torch::Tensor h3 = torch::bartlett_window(window_length);
            torch::Tensor h4 = torch::blackman_window(window_length);
            torch::Tensor h5 = torch::kaiser_window(window_length);
            (void)h1; (void)h2; (void)h3; (void)h4; (void)h5;
        } catch (const c10::Error& e) {
            // Expected errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}