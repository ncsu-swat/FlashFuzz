#include "fuzzer_utils.h" // General fuzzing utilities
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>       // For cerr
#include <optional>
#include <string>

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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor - rfft2 requires real-valued input
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // rfft2 requires at least 1D tensor and real dtype
        if (input.dim() < 1) {
            return 0;
        }
        
        // Convert to real dtype if complex
        if (input.is_complex()) {
            input = torch::real(input);
        }
        
        // Ensure float type for FFT
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Parse optional parameters if we have more data
        int64_t s_val = -1;            // Default: use input size
        int64_t dim1 = -2, dim2 = -1;  // Default dimensions for rfft2
        c10::optional<c10::string_view> norm = c10::nullopt;
        
        // Parse dimensions if we have more data
        if (offset + 2 <= Size) {
            uint8_t dim_selector1 = Data[offset++];
            uint8_t dim_selector2 = Data[offset++];
            
            // Map to valid dimension range
            if (input.dim() >= 2) {
                dim1 = (static_cast<int64_t>(dim_selector1) % input.dim()) - input.dim();
                dim2 = (static_cast<int64_t>(dim_selector2) % input.dim()) - input.dim();
                // Ensure dim1 != dim2
                if (dim1 == dim2) {
                    dim2 = (dim2 + 1) % input.dim() - input.dim();
                    if (dim1 == dim2) {
                        dim2 = dim1 - 1;
                        if (dim2 < -input.dim()) {
                            dim2 = -1;
                            dim1 = -2;
                        }
                    }
                }
            } else {
                // 1D tensor - can only do 1D FFT dimension
                dim1 = 0;
                dim2 = 0; // Will cause error, but that's expected
            }
        }
        
        // Parse s parameter if we have more data
        if (offset + sizeof(int32_t) <= Size) {
            int32_t s_raw;
            std::memcpy(&s_raw, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            
            // Bound positive sizes to keep allocations in check
            if (s_raw > 0) {
                constexpr int64_t max_fft_size = 64;
                s_val = std::max<int64_t>(1, std::min<int64_t>(s_raw, max_fft_size));
            }
        }
        
        // Parse norm parameter if we have more data
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++];
            switch (norm_selector % 4) {
                case 0:
                    norm = "backward";
                    break;
                case 1:
                    norm = "forward";
                    break;
                case 2:
                    norm = "ortho";
                    break;
                default:
                    norm = c10::nullopt;
                    break;
            }
        }
        
        // Prepare dimensions array
        std::array<int64_t, 2> dim_arr = {dim1, dim2};
        at::IntArrayRef dim_ref(dim_arr);
        
        // Prepare optional size parameter
        c10::optional<at::IntArrayRef> s_opt = c10::nullopt;
        std::array<int64_t, 2> s_arr;
        if (s_val > 0) {
            s_arr = {s_val, s_val};
            s_opt = at::IntArrayRef(s_arr);
        }
        
        // Apply rfft2 operation - this is the main API we're testing
        torch::Tensor output;
        try {
            output = torch::fft::rfft2(input, s_opt, dim_ref, norm);
        } catch (const c10::Error&) {
            // Expected for invalid dimension combinations
            return 0;
        }
        
        // Verify output is complex (rfft2 returns complex tensor)
        if (!output.is_complex()) {
            throw std::runtime_error("rfft2 should return complex tensor");
        }
        
        // Try additional operations on the output to exercise more code paths
        torch::Tensor abs_output = output.abs();
        torch::Tensor angle_output = torch::angle(output);
        
        // Try inverse operation with matching s parameter
        // For irfft2, we need to provide the original size to reconstruct properly
        try {
            std::array<int64_t, 2> orig_s_arr;
            if (s_val > 0) {
                orig_s_arr = {s_val, s_val};
            } else {
                // Use original input sizes for the transformed dimensions
                int64_t actual_dim1 = dim1 < 0 ? input.dim() + dim1 : dim1;
                int64_t actual_dim2 = dim2 < 0 ? input.dim() + dim2 : dim2;
                if (actual_dim1 >= 0 && actual_dim1 < input.dim() &&
                    actual_dim2 >= 0 && actual_dim2 < input.dim()) {
                    orig_s_arr = {input.size(actual_dim1), input.size(actual_dim2)};
                } else {
                    orig_s_arr = {input.size(-2), input.size(-1)};
                }
            }
            c10::optional<at::IntArrayRef> orig_s_opt = at::IntArrayRef(orig_s_arr);
            torch::Tensor reconstructed = torch::fft::irfft2(output, orig_s_opt, dim_ref, norm);
        } catch (const c10::Error&) {
            // irfft2 might fail for various reasons, that's OK
        }
        
        // Test with default parameters (no s, default dims)
        if (input.dim() >= 2) {
            try {
                torch::Tensor output_default = torch::fft::rfft2(input);
            } catch (const c10::Error&) {
                // May fail for edge cases
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