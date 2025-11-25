#include "fuzzer_utils.h"
#include <iostream>
#include <vector>
#include <optional>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // 1. Create the input tensor from fuzz data
        // This handles parsing random shapes, ranks, dtypes, and data elements.
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);

        // 2. Ensure we have enough data left for parameter selection
        // If not, fallback to default global mean
        if (offset >= Size)
        {
            torch::mean(input);
            return 0;
        }

        // 3. Parse Control Byte
        // We use a byte to decide which overload and optional arguments to use.
        uint8_t control = Data[offset++];
        
        // Bit 0: Mode Selection
        // 0 -> Global reduction: mean(input, *, dtype)
        // 1 -> Dimension reduction: mean(input, dim, keepdim, *, dtype)
        bool use_dim_reduction = (control & 0x01);

        // Bit 1: Provide explicit dtype?
        bool provide_dtype = (control & 0x02);
        
        // Bit 2: Keepdim (only relevant for dim reduction path)
        bool keepdim = (control & 0x04);

        // 4. Parse 'dtype' argument if requested
        std::optional<torch::ScalarType> dtype = std::nullopt;
        if (provide_dtype)
        {
            if (offset >= Size) return 0;
            dtype = fuzzer_utils::parseDataType(Data[offset++]);
        }

        if (use_dim_reduction)
        {
            // --- Dimension Reduction Path ---
            // Target: torch.mean(input, dim, keepdim, *, dtype)
            
            // Parse dimensions from the remaining fuzz data
            std::vector<int64_t> dims;
            
            if (offset < Size)
            {
                uint8_t dim_params = Data[offset++];
                int64_t rank = input.dim();
                
                // Determine how many dimensions to reduce over.
                // Using modulo to map the random byte to a reasonable range [0, rank+1]
                // (rank+1 allows us to test duplicate or slightly excessive dims)
                int num_dims_to_pick = 0;
                if (rank == 0) 
                {
                    // For scalar tensors, only empty list is strictly valid, 
                    // but we fuzz 0 or 1 dimensions to hit error paths.
                    num_dims_to_pick = dim_params % 2; 
                } 
                else 
                {
                    num_dims_to_pick = dim_params % (rank + 2);
                }

                for (int i = 0; i < num_dims_to_pick; ++i)
                {
                    if (offset >= Size) break;
                    uint8_t val = Data[offset++];
                    
                    int64_t d;
                    if (rank > 0) 
                    {
                        // Map byte to dimension index.
                        // 200/256 chance to be valid-ish (modulo rank)
                        // 56/256 chance to be raw (likely out of bounds)
                        if (val < 200) 
                        {
                            d = val % rank;
                            // Apply negation for negative indexing 50% of time (odd/even check)
                            if (val % 2 != 0) 
                            {
                                d -= rank;
                            }
                        } 
                        else 
                        {
                            d = val; // Raw value, likely OOB
                        }
                    } 
                    else 
                    {
                        d = val; // Scalar tensor, any non-empty dim is usually invalid
                    }
                    dims.push_back(d);
                }
            }

            // Execute the operation
            if (dtype.has_value())
            {
                torch::mean(input, dims, keepdim, dtype.value());
            }
            else
            {
                torch::mean(input, dims, keepdim);
            }
        }
        else
        {
            // --- Global Reduction Path ---
            // Target: torch.mean(input, *, dtype)
            
            if (dtype.has_value())
            {
                torch::mean(input, dtype.value());
            }
            else
            {
                torch::mean(input);
            }
        }
    }
    catch (const std::exception &e)
    {
        // Catch exceptions to prevent the fuzzer from aborting.
        // Expected exceptions include:
        // - "input must be floating point or complex" (if input is Int and no dtype provided)
        // - "Dimension out of range"
        // - "Could not infer output dtype"
        std::cout << "Exception caught: " << e.what() << std::endl;
        return 0; // Discard input but keep fuzzing
    }

    return 0;
}