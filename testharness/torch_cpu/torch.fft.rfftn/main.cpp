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
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse FFT dimensions if there's data left
        std::vector<int64_t> dim;
        if (offset + 1 < Size) {
            uint8_t num_dims = Data[offset++] % 5; // Up to 4 dimensions
            
            for (uint8_t i = 0; i < num_dims && offset < Size; i++) {
                int64_t d = static_cast<int64_t>(Data[offset++]);
                dim.push_back(d);
            }
        }
        
        // Parse norm parameter if there's data left
        std::optional<std::string> norm = std::nullopt;
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++] % 4;
            if (norm_selector == 0) {
                norm = "backward";
            } else if (norm_selector == 1) {
                norm = "forward";
            } else if (norm_selector == 2) {
                norm = "ortho";
            }
        }
        
        // Apply rfftn operation
        torch::Tensor output;
        if (dim.empty()) {
            if (norm.has_value()) {
                output = torch::fft::rfftn(input_tensor, c10::nullopt, c10::nullopt, norm.value());
            } else {
                output = torch::fft::rfftn(input_tensor);
            }
        } else {
            if (norm.has_value()) {
                output = torch::fft::rfftn(input_tensor, c10::nullopt, dim, norm.value());
            } else {
                output = torch::fft::rfftn(input_tensor, c10::nullopt, dim);
            }
        }
        
        // Ensure the output is used to prevent optimization
        if (output.defined()) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
