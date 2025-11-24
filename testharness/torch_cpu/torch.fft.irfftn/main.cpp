#include "fuzzer_utils.h" // General fuzzing utilities
#include <c10/util/Optional.h>
#include <cmath>
#include <cstring>
#include <iostream> // For cerr
#include <string>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse dimensions for irfftn
        std::vector<int64_t> dims;
        if (offset + 1 < Size) {
            uint8_t num_dims = Data[offset++] % 5; // Up to 4 dimensions

            for (uint8_t i = 0; i < num_dims && offset + sizeof(int64_t) <= Size; ++i) {
                int64_t dim;
                std::memcpy(&dim, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                dims.push_back(dim);
            }
        }
        
        // Parse norm parameter
        std::string norm = "backward";
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++] % 4;
            switch (norm_selector) {
                case 0: norm = "backward"; break;
                case 1: norm = "forward"; break;
                case 2: norm = "ortho"; break;
                default: norm = "backward";
            }
        }
        
        // Parse s parameter (output signal size)
        std::vector<int64_t> s_values;
        bool use_s = false;
        if (offset < Size && Data[offset++] % 2 == 0) { // 50% chance to use s parameter
            use_s = true;
            uint8_t s_size = (offset < Size) ? (Data[offset++] % 5) : 0; // Up to 4 dimensions

            for (uint8_t i = 0; i < s_size && offset + sizeof(int64_t) <= Size; ++i) {
                int64_t dim_size;
                std::memcpy(&dim_size, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);

                // Keep lengths small and positive to avoid huge allocations
                int64_t bounded = 1 + static_cast<int64_t>(std::abs(dim_size) % 16);
                s_values.push_back(bounded);
            }
        }

        // Ensure the input is complex as required by irfftn
        if (!input.is_complex()) {
            input = input.to(torch::kComplexFloat);
        }

        // Clamp dims to the input rank to avoid invalid axes
        const auto input_dim = input.dim();
        if (input_dim == 0) {
            dims.clear();
        } else {
            for (auto &d : dims) {
                int64_t wrapped = static_cast<int64_t>(std::abs(d)) % input_dim;
                d = wrapped;
            }
        }

        c10::optional<torch::IntArrayRef> dims_opt;
        if (!dims.empty() && input_dim > 0) {
            dims_opt = torch::IntArrayRef(dims);
        }

        c10::optional<torch::IntArrayRef> s_opt;
        if (use_s && !s_values.empty()) {
            s_opt = torch::IntArrayRef(s_values);
        }
        
        // Apply irfftn operation
        torch::Tensor result = torch::fft::irfftn(input, s_opt, dims_opt, norm);
        
        // Access result to ensure computation is performed
        auto sum = result.sum().item<double>();
        (void)sum; // Prevent unused variable warning
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
