#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Basic size check to ensure we can read at least some metadata
        if (Size < 5)
        {
            return 0;
        }

        // 1. Create the 'input' Tensor
        // This consumes bytes from Data based on the encoded rank and shape.
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);

        // 2. Parse Control Flags and Alpha
        // We need at least 1 byte for control and 4 bytes for alpha value
        if (offset + 5 > Size)
        {
            return 0;
        }

        uint8_t control_byte = Data[offset++];
        
        // Control bits:
        // Bit 0: 'other' is Tensor (1) or Scalar (0)
        bool other_is_tensor = (control_byte & 0x01);
        // Bit 1: Use the 'out' parameter variant (1) or functional (0)
        bool use_out = (control_byte & 0x02);
        // Bit 2: 'alpha' type is Int (1) or Float (0)
        bool alpha_is_int = (control_byte & 0x04);

        // Parse 'alpha' (Scalar)
        torch::Scalar alpha = 1.0;
        float raw_alpha_val;
        std::memcpy(&raw_alpha_val, Data + offset, sizeof(float));
        offset += sizeof(float);

        if (alpha_is_int) {
            alpha = static_cast<int>(raw_alpha_val);
        } else {
            alpha = raw_alpha_val;
        }

        // 3. Execute torch::add variants
        if (other_is_tensor)
        {
            // Case A: torch.add(Tensor, Tensor, alpha=...)
            // Create 'other' Tensor
            // If createTensor fails due to lack of data, it throws runtime_error which is caught.
            torch::Tensor other = fuzzer_utils::createTensor(Data, Size, offset);

            if (use_out)
            {
                // Variant: with 'out' parameter
                torch::Tensor out = fuzzer_utils::createTensor(Data, Size, offset);
                torch::add_out(out, input, other, alpha);
            }
            else
            {
                // Variant: functional
                torch::add(input, other, alpha);
            }
        }
        else
        {
            // Case B: torch.add(Tensor, Number, alpha=...)
            // Parse 'other' as Scalar
            torch::Scalar other_scalar = 1.0;
            
            if (offset + sizeof(float) <= Size)
            {
                float raw_other;
                std::memcpy(&raw_other, Data + offset, sizeof(float));
                offset += sizeof(float);
                other_scalar = raw_other;
            }
            // If not enough data for scalar, we rely on the default initialized value 1.0

            if (use_out)
            {
                // Variant: with 'out' parameter
                torch::Tensor out = fuzzer_utils::createTensor(Data, Size, offset);
                torch::add_out(out, input, other_scalar, alpha);
            }
            else
            {
                // Variant: functional
                torch::add(input, other_scalar, alpha);
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return 0; // keep the input
    }

    return 0;
}