#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least enough data for tensor creation and axis parameters
        if (Size < 16) {
            return 0;
        }

        // Generate tensor dimensions (1-6 dimensions)
        int num_dims = (Data[offset] % 6) + 1;
        offset++;

        std::vector<int64_t> dims;
        for (int i = 0; i < num_dims && offset < Size; i++) {
            int64_t dim_size = (Data[offset] % 10) + 1; // 1-10 size per dimension
            dims.push_back(dim_size);
            offset++;
        }

        if (dims.empty()) {
            dims.push_back(1); // Fallback to 1D tensor
        }

        // Create input tensor with random data
        torch::Tensor input = torch::randn(dims);

        // Test different data types occasionally
        if (offset < Size && Data[offset] % 4 == 0) {
            input = input.to(torch::kFloat64);
        } else if (offset < Size && Data[offset] % 4 == 1) {
            input = input.to(torch::kInt32);
        } else if (offset < Size && Data[offset] % 4 == 2) {
            input = input.to(torch::kInt64);
        }
        if (offset < Size) offset++;

        // Generate source and destination axes
        if (offset + 1 >= Size) {
            return 0;
        }

        int64_t source_axis = static_cast<int64_t>(static_cast<int8_t>(Data[offset])) % static_cast<int64_t>(dims.size());
        offset++;
        int64_t dest_axis = static_cast<int64_t>(static_cast<int8_t>(Data[offset + 1])) % static_cast<int64_t>(dims.size());
        offset++;

        // Test single axis moveaxis
        torch::Tensor result1 = torch::moveaxis(input, source_axis, dest_axis);

        // Test with negative indices
        int64_t neg_source = source_axis - static_cast<int64_t>(dims.size());
        int64_t neg_dest = dest_axis - static_cast<int64_t>(dims.size());
        torch::Tensor result2 = torch::moveaxis(input, neg_source, neg_dest);

        // Test multiple axes moveaxis if we have enough dimensions and data
        if (dims.size() >= 2 && offset + 3 < Size) {
            std::vector<int64_t> source_axes;
            std::vector<int64_t> dest_axes;
            
            int num_axes = std::min(static_cast<int>(dims.size()), (Data[offset] % 3) + 1);
            offset++;
            
            for (int i = 0; i < num_axes && offset < Size; i++) {
                int64_t src = static_cast<int64_t>(static_cast<int8_t>(Data[offset])) % static_cast<int64_t>(dims.size());
                source_axes.push_back(src);
                offset++;
                
                if (offset < Size) {
                    int64_t dst = static_cast<int64_t>(static_cast<int8_t>(Data[offset])) % static_cast<int64_t>(dims.size());
                    dest_axes.push_back(dst);
                    offset++;
                }
            }
            
            if (source_axes.size() == dest_axes.size() && !source_axes.empty()) {
                torch::Tensor result3 = torch::moveaxis(input, source_axes, dest_axes);
            }
        }

        // Test edge cases with boundary values
        if (dims.size() > 1) {
            // Move first axis to last
            torch::Tensor result4 = torch::moveaxis(input, 0, static_cast<int64_t>(dims.size()) - 1);
            
            // Move last axis to first
            torch::Tensor result5 = torch::moveaxis(input, static_cast<int64_t>(dims.size()) - 1, 0);
            
            // Move axis to same position (should be no-op)
            torch::Tensor result6 = torch::moveaxis(input, 0, 0);
        }

        // Test with different tensor layouts if possible
        if (offset < Size && Data[offset] % 3 == 0 && input.numel() > 1) {
            torch::Tensor contiguous_input = input.contiguous();
            torch::Tensor result7 = torch::moveaxis(contiguous_input, source_axis, dest_axis);
        }

        // Test with sliced/viewed tensors
        if (offset < Size && Data[offset] % 5 == 0 && dims.size() > 0 && dims[0] > 1) {
            torch::Tensor sliced = input.slice(0, 0, dims[0] / 2);
            torch::Tensor result8 = torch::moveaxis(sliced, 0, (sliced.dim() > 1) ? 1 : 0);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}