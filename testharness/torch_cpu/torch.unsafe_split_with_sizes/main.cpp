#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <numeric>        // For std::accumulate

// --- Fuzzer Entry Point ---
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
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip 0-dim tensors as split doesn't make sense for scalars
        if (input_tensor.dim() == 0) {
            return 0;
        }
        
        // Parse dimension to split along
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make sure dim is within valid range
            dim = dim % input_tensor.dim();
            if (dim < 0) {
                dim += input_tensor.dim();
            }
        }
        
        // Get the size along the split dimension
        int64_t dim_size = input_tensor.size(dim);
        
        // Skip if dimension size is 0
        if (dim_size == 0) {
            return 0;
        }
        
        // Parse number of sections (at least 1, at most dim_size)
        uint8_t num_sections = 1;
        if (offset < Size) {
            num_sections = Data[offset++];
            num_sections = std::max(num_sections, static_cast<uint8_t>(1));
            // Limit to reasonable number and not more than dim_size
            num_sections = std::min(num_sections, static_cast<uint8_t>(std::min(static_cast<int64_t>(16), dim_size)));
        }
        
        // Create section sizes vector that sums to dim_size
        std::vector<int64_t> section_sizes;
        int64_t remaining = dim_size;
        
        for (uint8_t i = 0; i < num_sections - 1; ++i) {
            int64_t size_val = 1;
            if (offset < Size && remaining > (num_sections - i)) {
                // Use fuzzer data to determine split point
                uint8_t fuzz_byte = Data[offset++];
                // Distribute remaining among this and subsequent sections
                int64_t max_for_this = remaining - (num_sections - i - 1); // Leave at least 1 for each remaining
                size_val = 1 + (fuzz_byte % std::min(static_cast<int64_t>(255), max_for_this));
                size_val = std::min(size_val, max_for_this);
            }
            section_sizes.push_back(size_val);
            remaining -= size_val;
        }
        // Last section gets the remainder
        section_sizes.push_back(remaining);
        
        // Verify sizes sum correctly (defensive check)
        int64_t total = std::accumulate(section_sizes.begin(), section_sizes.end(), static_cast<int64_t>(0));
        if (total != dim_size) {
            return 0; // Skip malformed input
        }
        
        // Apply the unsafe_split_with_sizes operation
        std::vector<torch::Tensor> result = torch::unsafe_split_with_sizes(input_tensor, section_sizes, dim);
        
        // Perform some basic operations on the result to ensure it's used
        if (!result.empty()) {
            // Verify we got the expected number of tensors
            if (result.size() != section_sizes.size()) {
                std::cerr << "Unexpected result size" << std::endl;
            }
            
            // Access each resulting tensor to exercise the API fully
            for (size_t i = 0; i < result.size(); ++i) {
                // Force materialization by accessing data
                volatile float sum_val = result[i].sum().item<float>();
                (void)sum_val;
            }
        }
        
        // Also test with contiguous tensor
        if (offset < Size && (Data[offset] % 2 == 0)) {
            torch::Tensor contiguous_input = input_tensor.contiguous();
            std::vector<torch::Tensor> result2 = torch::unsafe_split_with_sizes(contiguous_input, section_sizes, dim);
            for (const auto& t : result2) {
                volatile float v = t.sum().item<float>();
                (void)v;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}