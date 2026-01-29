#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Create input tensor - need at least some data
        if (Size < 4) return 0;
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // FFT requires at least 1D tensor
        if (input_tensor.dim() == 0) {
            input_tensor = input_tensor.unsqueeze(0);
        }
        
        // Parse FFT dimensions
        std::vector<int64_t> dim;
        if (offset + 1 < Size && input_tensor.dim() > 0) {
            uint8_t num_dims = Data[offset++] % std::min(static_cast<int64_t>(4), input_tensor.dim());
            num_dims = std::min(num_dims, static_cast<uint8_t>(input_tensor.dim()));
            
            std::set<int64_t> used_dims;
            for (uint8_t i = 0; i < num_dims && offset < Size; i++) {
                int64_t d = static_cast<int64_t>(Data[offset++]) % input_tensor.dim();
                // Avoid duplicate dimensions
                if (used_dims.find(d) == used_dims.end()) {
                    dim.push_back(d);
                    used_dims.insert(d);
                }
            }
        }
        
        // Parse normalization mode
        std::string norm = "backward";
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++] % 3;
            if (norm_selector == 0) norm = "backward";
            else if (norm_selector == 1) norm = "ortho";
            else norm = "forward";
        }
        
        // Parse s parameter (optional output size) - must match dim size if both provided
        std::optional<std::vector<int64_t>> s = std::nullopt;
        if (offset < Size && Data[offset++] % 2 == 0) {
            std::vector<int64_t> s_vec;
            // s size should match dim size, or be based on input dims if dim is empty
            size_t target_size = dim.empty() ? input_tensor.dim() : dim.size();
            target_size = std::min(target_size, static_cast<size_t>(4));
            
            for (size_t i = 0; i < target_size && offset < Size; i++) {
                int64_t size_val = static_cast<int64_t>(Data[offset++] % 32) + 1; // Ensure positive
                s_vec.push_back(size_val);
            }
            
            if (!s_vec.empty() && (dim.empty() || s_vec.size() == dim.size())) {
                s = s_vec;
            }
        }
        
        // Inner try-catch for expected runtime errors (shape mismatches, etc.)
        try {
            // Apply ifftn operation
            torch::Tensor result;
            if (dim.empty()) {
                if (s.has_value()) {
                    result = torch::fft::ifftn(input_tensor, s.value(), c10::nullopt, norm);
                } else {
                    result = torch::fft::ifftn(input_tensor, c10::nullopt, c10::nullopt, norm);
                }
            } else {
                if (s.has_value()) {
                    result = torch::fft::ifftn(input_tensor, s.value(), dim, norm);
                } else {
                    result = torch::fft::ifftn(input_tensor, c10::nullopt, dim, norm);
                }
            }
            
            // Force computation to ensure any errors are triggered
            result.sum().item<double>();
        }
        catch (const c10::Error&) {
            // Expected errors from invalid parameter combinations - ignore silently
        }
        catch (const std::runtime_error&) {
            // Expected runtime errors - ignore silently
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}