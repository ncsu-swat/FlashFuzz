#include "fuzzer_utils.h" // General fuzzing utilities
#include <c10/util/Optional.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <set>

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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        const auto input_dim = input.dim();
        if (input_dim == 0) {
            return 0; // Skip scalar tensors
        }
        
        // Parse dimensions for irfftn - ensure unique dimensions
        std::vector<int64_t> dims;
        std::set<int64_t> dims_set;
        if (offset + 1 < Size) {
            uint8_t num_dims = (Data[offset++] % std::min(static_cast<int64_t>(4), input_dim)) + 1;

            for (uint8_t i = 0; i < num_dims && offset < Size; ++i) {
                int64_t dim = static_cast<int64_t>(Data[offset++] % input_dim);
                // Ensure unique dimensions
                if (dims_set.find(dim) == dims_set.end()) {
                    dims_set.insert(dim);
                    dims.push_back(dim);
                }
            }
        }
        
        if (dims.empty()) {
            // Default to last dimension if no valid dims parsed
            dims.push_back(input_dim - 1);
        }
        
        // Parse norm parameter
        std::string norm = "backward";
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++] % 3;
            switch (norm_selector) {
                case 0: norm = "backward"; break;
                case 1: norm = "forward"; break;
                case 2: norm = "ortho"; break;
            }
        }
        
        // Parse s parameter (output signal size) - must match dims length if specified
        std::vector<int64_t> s_values;
        bool use_s = false;
        if (offset < Size && Data[offset++] % 2 == 0) {
            use_s = true;
            // s must have same length as dims
            for (size_t i = 0; i < dims.size() && offset < Size; ++i) {
                int64_t dim_size = static_cast<int64_t>(Data[offset++] % 15) + 2; // 2-16
                s_values.push_back(dim_size);
            }
            // If we couldn't fill s_values to match dims, don't use s
            if (s_values.size() != dims.size()) {
                use_s = false;
                s_values.clear();
            }
        }

        // Ensure the input is complex as required by irfftn
        if (!input.is_complex()) {
            input = input.to(torch::kComplexFloat);
        }

        c10::optional<torch::IntArrayRef> dims_opt = torch::IntArrayRef(dims);

        c10::optional<torch::IntArrayRef> s_opt;
        if (use_s && !s_values.empty()) {
            s_opt = torch::IntArrayRef(s_values);
        }
        
        // Apply irfftn operation - inner try-catch for expected failures
        torch::Tensor result;
        try {
            result = torch::fft::irfftn(input, s_opt, dims_opt, norm);
        } catch (const c10::Error &e) {
            // Expected failures (shape mismatches, etc.) - silently discard
            return 0;
        }
        
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