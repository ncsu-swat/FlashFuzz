#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

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
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor - FFT requires float or complex types
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is at least 2D for fft2
        if (input.dim() < 2) {
            input = input.unsqueeze(0).unsqueeze(0);
        } else if (input.dim() == 1) {
            input = input.unsqueeze(0);
        }
        
        // Convert to float if needed (FFT requires floating point)
        if (!input.is_floating_point() && !input.is_complex()) {
            input = input.to(torch::kFloat32);
        }
        
        // Parse FFT2 parameters if we have more data
        int64_t n_h = -1;
        int64_t n_w = -1;
        int64_t dim_h = -2;
        int64_t dim_w = -1;
        uint8_t norm_mode = 0;
        
        // Parse n_h and n_w (output size) - limit to reasonable values to avoid OOM
        if (offset + sizeof(int32_t) <= Size) {
            int32_t tmp;
            std::memcpy(&tmp, Data + offset, sizeof(int32_t));
            n_h = (tmp % 256) + 1; // Limit to 1-256
            offset += sizeof(int32_t);
        }
        
        if (offset + sizeof(int32_t) <= Size) {
            int32_t tmp;
            std::memcpy(&tmp, Data + offset, sizeof(int32_t));
            n_w = (tmp % 256) + 1; // Limit to 1-256
            offset += sizeof(int32_t);
        }
        
        // Parse dim indices
        if (offset + 2 <= Size) {
            int8_t d1 = static_cast<int8_t>(Data[offset++]);
            int8_t d2 = static_cast<int8_t>(Data[offset++]);
            
            // Normalize dimensions to valid range
            int64_t ndim = input.dim();
            dim_h = ((d1 % ndim) + ndim) % ndim;
            dim_w = ((d2 % ndim) + ndim) % ndim;
            
            // Ensure different dimensions
            if (dim_h == dim_w) {
                dim_w = (dim_h + 1) % ndim;
            }
        }
        
        // Parse normalization mode
        if (offset < Size) {
            norm_mode = Data[offset++] % 4;
        }
        
        // Determine normalization string
        c10::optional<c10::string_view> norm_opt = c10::nullopt;
        if (norm_mode == 1) {
            norm_opt = "forward";
        } else if (norm_mode == 2) {
            norm_opt = "backward";
        } else if (norm_mode == 3) {
            norm_opt = "ortho";
        }
        
        // Select test case based on remaining data
        uint8_t test_case = (offset < Size) ? (Data[offset++] % 5) : 0;
        
        torch::Tensor output;
        
        try {
            switch (test_case) {
                case 0:
                    // Basic FFT2 with no parameters
                    output = torch::fft::fft2(input);
                    break;
                    
                case 1:
                    // FFT2 with specified output size
                    output = torch::fft::fft2(input, std::vector<int64_t>{n_h, n_w});
                    break;
                    
                case 2:
                    // FFT2 with specified dimensions
                    output = torch::fft::fft2(input, c10::nullopt, std::vector<int64_t>{dim_h, dim_w});
                    break;
                    
                case 3:
                    // FFT2 with normalization
                    output = torch::fft::fft2(input, c10::nullopt, std::vector<int64_t>{-2, -1}, norm_opt);
                    break;
                    
                case 4:
                    // FFT2 with all parameters
                    output = torch::fft::fft2(input, std::vector<int64_t>{n_h, n_w}, 
                                              std::vector<int64_t>{dim_h, dim_w}, norm_opt);
                    break;
            }
            
            // Force evaluation of the output tensor
            auto sum = output.sum();
            (void)sum.item<float>();
            
            // Try inverse FFT2 to check round-trip on some inputs
            if (test_case % 2 == 0) {
                auto inverse = torch::fft::ifft2(output, c10::nullopt, 
                                                  std::vector<int64_t>{-2, -1}, norm_opt);
                auto inverse_sum = inverse.sum();
                (void)inverse_sum.item<float>();
            }
            
            // Also test rfft2/irfft2 for real inputs
            if (test_case == 0 && !input.is_complex()) {
                auto rfft_out = torch::fft::rfft2(input);
                auto rfft_sum = rfft_out.sum();
                (void)rfft_sum.item<float>();
            }
            
        } catch (const c10::Error& e) {
            // Expected errors from invalid parameter combinations - silently ignore
        } catch (const std::runtime_error& e) {
            // Shape/dimension mismatches are expected - silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}