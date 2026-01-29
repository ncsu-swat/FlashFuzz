#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

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
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor - rfftn expects real input
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // rfftn requires real-valued input
        if (input_tensor.is_complex()) {
            input_tensor = torch::real(input_tensor);
        }
        
        // Ensure tensor is floating point for FFT
        if (!input_tensor.is_floating_point()) {
            input_tensor = input_tensor.to(torch::kFloat32);
        }
        
        int64_t ndim = input_tensor.dim();
        if (ndim == 0) {
            // FFT doesn't work on 0-dimensional tensors
            return 0;
        }
        
        // Parse FFT dimensions if there's data left
        std::vector<int64_t> dim_vec;
        bool has_dim = false;
        if (offset + 1 < Size) {
            uint8_t num_dims = Data[offset++] % std::min(static_cast<int64_t>(4), ndim + 1);
            
            if (num_dims > 0) {
                for (uint8_t i = 0; i < num_dims && offset < Size; i++) {
                    // Map to valid dimension indices
                    int64_t d = static_cast<int64_t>(Data[offset++]) % ndim;
                    // Use negative indexing sometimes for variety
                    if (offset < Size && Data[offset - 1] % 2 == 0) {
                        d = d - ndim;
                    }
                    dim_vec.push_back(d);
                }
                if (!dim_vec.empty()) {
                    has_dim = true;
                }
            }
        }
        
        // Parse output sizes 's' parameter if there's data left
        std::vector<int64_t> s_vec;
        bool has_s = false;
        if (offset + 1 < Size && has_dim) {
            uint8_t use_s = Data[offset++] % 3;
            if (use_s == 1) {
                for (size_t i = 0; i < dim_vec.size() && offset < Size; i++) {
                    // Create reasonable output sizes (1 to 32)
                    int64_t size = (static_cast<int64_t>(Data[offset++]) % 32) + 1;
                    s_vec.push_back(size);
                }
                if (s_vec.size() == dim_vec.size()) {
                    has_s = true;
                }
            }
        }
        
        // Parse norm parameter if there's data left
        std::optional<c10::string_view> norm = std::nullopt;
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++] % 4;
            if (norm_selector == 0) {
                norm = "backward";
            } else if (norm_selector == 1) {
                norm = "forward";
            } else if (norm_selector == 2) {
                norm = "ortho";
            }
            // norm_selector == 3 means use default (nullopt)
        }
        
        // Apply rfftn operation with various parameter combinations
        torch::Tensor output;
        try {
            if (!has_dim) {
                // No dim specified - use defaults
                if (has_s) {
                    if (norm.has_value()) {
                        output = torch::fft::rfftn(input_tensor, s_vec, c10::nullopt, norm.value());
                    } else {
                        output = torch::fft::rfftn(input_tensor, s_vec);
                    }
                } else {
                    if (norm.has_value()) {
                        output = torch::fft::rfftn(input_tensor, c10::nullopt, c10::nullopt, norm.value());
                    } else {
                        output = torch::fft::rfftn(input_tensor);
                    }
                }
            } else {
                // dim specified
                if (has_s) {
                    if (norm.has_value()) {
                        output = torch::fft::rfftn(input_tensor, s_vec, dim_vec, norm.value());
                    } else {
                        output = torch::fft::rfftn(input_tensor, s_vec, dim_vec);
                    }
                } else {
                    if (norm.has_value()) {
                        output = torch::fft::rfftn(input_tensor, c10::nullopt, dim_vec, norm.value());
                    } else {
                        output = torch::fft::rfftn(input_tensor, c10::nullopt, dim_vec);
                    }
                }
            }
        } catch (const c10::Error&) {
            // Expected errors from invalid dimension combinations - silently ignore
            return 0;
        }
        
        // Ensure the output is used to prevent optimization
        // rfftn returns a complex tensor, so use abs() before sum
        if (output.defined() && output.numel() > 0) {
            volatile float sum = torch::abs(output).sum().item<float>();
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