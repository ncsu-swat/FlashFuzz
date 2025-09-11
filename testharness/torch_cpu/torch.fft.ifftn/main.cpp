#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Create input tensor
        if (offset >= Size) return 0;
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse FFT dimensions
        std::vector<int64_t> dim;
        if (offset + 1 < Size) {
            uint8_t num_dims = Data[offset++] % (input_tensor.dim() + 1);
            
            for (uint8_t i = 0; i < num_dims && offset < Size; i++) {
                int64_t d = static_cast<int64_t>(Data[offset++]) % input_tensor.dim();
                dim.push_back(d);
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
        
        // Parse s parameter (optional output size)
        std::optional<std::vector<int64_t>> s = std::nullopt;
        if (offset < Size && Data[offset++] % 2 == 0) {
            std::vector<int64_t> s_vec;
            uint8_t s_size = (offset < Size) ? Data[offset++] % 5 : 0;
            
            for (uint8_t i = 0; i < s_size && offset < Size; i++) {
                int64_t size_val;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&size_val, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                } else {
                    size_val = static_cast<int64_t>(Data[offset++]);
                }
                s_vec.push_back(std::abs(size_val) % 32);
            }
            
            if (!s_vec.empty()) {
                s = s_vec;
            }
        }
        
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
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
