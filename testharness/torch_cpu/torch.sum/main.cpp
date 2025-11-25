#include "fuzzer_utils.h"
#include <iostream>
#include <vector>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // 1. Create Input Tensor
        // Uses fuzzer_utils to generate a tensor with varied shape, rank, and dtype.
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);

        // Check if we have enough data left for control bits
        if (offset >= Size)
        {
            // With no extra data, just run the default sum on the tensor
            torch::sum(input);
            return 0;
        }

        // 2. Parse Control Byte
        // Bit 0: Select between global sum vs dim-specific sum
        // Bit 1: Specify output dtype
        // Bit 2: keepdim (only relevant for dim-specific sum)
        uint8_t control = Data[offset++];
        bool use_dim_args = control & 0x01;
        bool use_dtype = control & 0x02;
        bool keepdim = control & 0x04;

        // 3. Parse Optional Output Dtype
        c10::optional<torch::ScalarType> dtype = c10::nullopt;
        if (use_dtype)
        {
            if (offset < Size)
            {
                uint8_t dtype_selector = Data[offset++];
                dtype = fuzzer_utils::parseDataType(dtype_selector);
            }
        }

        // 4. Execute Target API
        if (!use_dim_args)
        {
            // Variant 1: sum(input, *, dtype=None)
            // Reduces all dimensions
            torch::sum(input, dtype);
        }
        else
        {
            // Variant 2: sum(input, dim, keepdim=False, *, dtype=None)
            std::vector<int64_t> dims;
            int64_t rank = input.dim();

            // Parse dimensions
            // We determine how many dimensions to reduce based on the next byte
            if (offset < Size)
            {
                uint8_t dim_count_byte = Data[offset++];
                
                // If rank > 0, we allow picking [0, rank] dimensions generally, 
                // but also allow fuzzing slightly larger lists or duplicates.
                size_t num_dims_to_pick = 0;
                if (rank > 0) 
                {
                    num_dims_to_pick = dim_count_byte % (rank + 2); 
                }
                else 
                {
                    // For rank 0, dimensions list should usually be empty.
                    // Randomly try to inject a dimension to test error handling.
                    if (dim_count_byte % 5 == 0) num_dims_to_pick = 1;
                }

                dims.reserve(num_dims_to_pick);
                for (size_t i = 0; i < num_dims_to_pick; ++i)
                {
                    if (offset >= Size) break;
                    uint8_t dim_byte = Data[offset++];

                    if (rank > 0)
                    {
                        // Convert byte to range [-rank, rank-1] to ensure valid indices
                        // while also testing negative indexing.
                        int64_t dim_val = (static_cast<int64_t>(dim_byte) % (rank * 2)) - rank;
                        dims.push_back(dim_val);
                    }
                    else
                    {
                        // For rank 0, any integer is invalid, but we fuzz it anyway.
                        dims.push_back(static_cast<int64_t>(dim_byte % 3)); 
                    }
                }
            }

            torch::sum(input, dims, keepdim, dtype);
        }
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; 
    }
    return 0;
}