#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse window parameters from the remaining data
        int64_t window_length = 10;
        
        if (offset + 2 < Size) {
            // Extract window_length from data
            uint16_t raw_window_length;
            std::memcpy(&raw_window_length, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
            window_length = static_cast<int64_t>(raw_window_length % 100) + 1; // Ensure positive
        }
        
        // Parse window function type
        std::string window_fn = "hann";
        if (offset < Size) {
            uint8_t window_fn_selector = Data[offset++] % 7;
            
            switch (window_fn_selector) {
                case 0:
                    window_fn = "hann";
                    break;
                case 1:
                    window_fn = "hamming";
                    break;
                case 2:
                    window_fn = "bartlett";
                    break;
                case 3:
                    window_fn = "blackman";
                    break;
                case 4:
                    window_fn = "kaiser";
                    break;
                case 5:
                    window_fn = "gaussian";
                    break;
                case 6:
                    window_fn = "tukey";
                    break;
            }
        }
        
        // Parse additional parameters for specific window functions
        double beta = 12.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&beta, Data + offset, sizeof(double));
            offset += sizeof(double);
            if (beta < 0) beta = 12.0;
        }
        
        // Parse additional options
        bool periodic = true;
        if (offset < Size) {
            periodic = Data[offset++] % 2 == 0;
        }
        
        // Try different window functions
        try {
            // Basic Hann window
            torch::Tensor result1 = torch::hann_window(window_length);
            
            // Hamming window
            torch::Tensor result2 = torch::hamming_window(window_length);
            
            // Bartlett window
            torch::Tensor result3 = torch::bartlett_window(window_length);
            
            // Blackman window
            torch::Tensor result4 = torch::blackman_window(window_length);
            
            // Kaiser window with beta parameter
            torch::Tensor result5 = torch::kaiser_window(window_length, periodic, beta);
            
            // Test with periodic parameter
            torch::Tensor result6 = torch::hann_window(window_length, periodic);
            torch::Tensor result7 = torch::hamming_window(window_length, periodic);
            torch::Tensor result8 = torch::bartlett_window(window_length, periodic);
            torch::Tensor result9 = torch::blackman_window(window_length, periodic);
            
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected and can be ignored
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}