#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // 1. Create Input Tensor
        // Parses dtype, rank, shape, and tensor data from the input byte stream.
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);

        // 2. Parse 'dim' argument
        // We need at least one byte remaining to determine the dimension.
        if (offset >= Size)
        {
            return 0;
        }

        uint8_t dim_byte = Data[offset++];
        int64_t dim = 0;
        int64_t rank = input.dim();

        // Map the random byte to a dimension index.
        // Valid range for dim is [-rank, rank - 1] (if rank > 0).
        // To ensure we fuzz edge cases, we map the input byte to a range 
        // slightly wider than the valid range: [-rank - 1, rank].
        if (rank == 0)
        {
            // Softmax on 0-dim tensors usually throws, but we fuzz simple values like -1, 0, 1.
            dim = (static_cast<int64_t>(dim_byte) % 3) - 1; 
        }
        else
        {
            // Example for Rank 2 (valid: -2, -1, 0, 1).
            // Target range: [-3, 2]. Total indices = 6.
            // Formula: range_width = 2 * rank + 2.
            // Offset shift: -(rank + 1).
            int64_t range_width = 2 * rank + 2;
            dim = (static_cast<int64_t>(dim_byte) % range_width) - (rank + 1);
        }

        // 3. Parse 'dtype' argument (optional)
        // torch::softmax(..., *, dtype=None)
        c10::optional<torch::ScalarType> dtype = c10::nullopt;
        
        if (offset < Size)
        {
            uint8_t dtype_byte = Data[offset++];
            // Use the MSB to decide if we should provide a dtype.
            // This gives ~50% chance of using the default (None).
            if (dtype_byte & 0x80)
            {
                // Use lower bits to pick a specific supported type.
                dtype = fuzzer_utils::parseDataType(dtype_byte & 0x7F);
            }
        }

        // 4. Invoke the Operation
        torch::softmax(input, dim, dtype);

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input for libfuzzer
    }
    return 0; // keep the input
}