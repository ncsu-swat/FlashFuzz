#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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

        if (Size < 4) {
            return 0;
        }

        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // hfftn expects complex input - convert to complex if real
        if (!input_tensor.is_complex()) {
            input_tensor = torch::complex(input_tensor, torch::zeros_like(input_tensor));
        }
        
        // Ensure tensor has at least 1 dimension
        if (input_tensor.dim() == 0) {
            input_tensor = input_tensor.unsqueeze(0);
        }

        int64_t ndim = input_tensor.dim();

        // Parse dim parameter first (dimensions to transform)
        std::vector<int64_t> dim;
        if (offset + 1 < Size) {
            uint8_t n_dims = (Data[offset++] % std::min<int64_t>(ndim, 3)) + 1;

            for (uint8_t i = 0; i < n_dims && offset < Size; i++) {
                // Map to valid dimension indices
                int64_t d = static_cast<int64_t>(Data[offset++]) % ndim;
                // Avoid duplicate dimensions
                bool duplicate = false;
                for (auto existing_d : dim) {
                    if (existing_d == d || existing_d == d - ndim) {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate) {
                    dim.push_back(d);
                }
            }
        }
        
        // If no valid dims, use last dimension
        if (dim.empty()) {
            dim.push_back(ndim - 1);
        }

        // Parse s parameter (output sizes for transformed dimensions)
        // s size should match dim size
        c10::optional<c10::IntArrayRef> s_opt = c10::nullopt;
        std::vector<int64_t> s;
        if (offset + 1 < Size) {
            uint8_t use_s = Data[offset++] % 2;
            if (use_s) {
                for (size_t i = 0; i < dim.size() && offset < Size; i++) {
                    // Use positive values for output sizes (1 to 64)
                    int64_t s_val = (static_cast<int64_t>(Data[offset++]) % 64) + 1;
                    s.push_back(s_val);
                }
                if (s.size() == dim.size()) {
                    s_opt = s;
                } else {
                    s.clear();
                }
            }
        }

        // Parse norm parameter
        c10::optional<c10::string_view> norm = c10::nullopt;
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++] % 4;
            if (norm_selector == 0) {
                norm = "backward";
            } else if (norm_selector == 1) {
                norm = "forward";
            } else if (norm_selector == 2) {
                norm = "ortho";
            }
            // norm_selector == 3 keeps norm as nullopt
        }

        // Call torch::fft::hfftn with proper parameter order: (input, s, dim, norm)
        // dim is required as IntArrayRef, not optional
        torch::Tensor result;
        
        try {
            result = torch::fft::hfftn(input_tensor, s_opt, dim, norm);
        } catch (const c10::Error &e) {
            // Shape mismatches and invalid dimension errors are expected
            return 0;
        }

        // Access result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            auto sum = result.sum();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}