#include "fuzzer_utils.h"
#include <iostream>
#include <algorithm>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least some data to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor - ihfft2 expects real-valued input
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input is real (not complex) and floating point
        if (input.is_complex()) {
            input = torch::real(input);
        }
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Ensure input has at least 2 dimensions for 2D FFT
        if (input.dim() < 2) {
            input = input.unsqueeze(0);
            if (input.dim() < 2) {
                input = input.unsqueeze(0);
            }
        }
        
        // Parse optional parameters if we have more data
        bool use_s = false;
        std::vector<int64_t> s_vec;
        std::vector<int64_t> dim = {-2, -1};
        c10::optional<c10::string_view> norm_opt = c10::nullopt;
        
        // Parse s parameter (output size)
        if (offset + 1 < Size) {
            use_s = (Data[offset++] % 2) == 1;
            if (use_s && offset + 2 * sizeof(int64_t) <= Size) {
                int64_t s_h, s_w;
                memcpy(&s_h, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                memcpy(&s_w, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Make sure s values are reasonable (positive, bounded)
                s_h = std::max<int64_t>(1, std::abs(s_h) % 100 + 1);
                s_w = std::max<int64_t>(1, std::abs(s_w) % 100 + 1);
                s_vec = {s_h, s_w};
            } else {
                use_s = false;
            }
        }
        
        // Parse dim parameter
        if (offset + 1 < Size) {
            uint8_t use_dim = Data[offset++] % 2;
            if (use_dim && offset + 2 * sizeof(int64_t) <= Size) {
                int64_t dim_h, dim_w;
                memcpy(&dim_h, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                memcpy(&dim_w, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Constrain dims to valid range for the input tensor
                int64_t ndim = input.dim();
                dim_h = dim_h % ndim;
                dim_w = dim_w % ndim;
                
                // Ensure dims are different
                if (dim_h == dim_w) {
                    dim_w = (dim_w + 1) % ndim;
                }
                
                dim = {dim_h, dim_w};
            }
        }
        
        // Parse norm parameter
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++] % 4;
            if (norm_selector == 1) {
                norm_opt = "backward";
            } else if (norm_selector == 2) {
                norm_opt = "ortho";
            } else if (norm_selector == 3) {
                norm_opt = "forward";
            }
            // norm_selector == 0 means use default (nullopt)
        }
        
        // Apply the ihfft2 operation
        torch::Tensor output;
        
        try {
            if (use_s) {
                output = torch::fft::ihfft2(input, c10::IntArrayRef(s_vec), dim, norm_opt);
            } else {
                output = torch::fft::ihfft2(input, c10::nullopt, dim, norm_opt);
            }
        } catch (const c10::Error &e) {
            // Expected failures due to invalid parameter combinations
            return 0;
        }
        
        // Perform some operation on the output to ensure it's used
        // Use abs() since output is complex
        auto sum = torch::abs(output).sum();
        
        // Touch the result to prevent optimization
        volatile float result = sum.item<float>();
        (void)result;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}