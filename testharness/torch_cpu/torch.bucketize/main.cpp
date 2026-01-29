#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <algorithm>      // For std::sort

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
        
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create boundaries tensor - must be 1D and sorted
        torch::Tensor boundaries_raw = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Flatten boundaries to 1D and sort (bucketize requires sorted 1D boundaries)
        torch::Tensor boundaries = std::get<0>(boundaries_raw.flatten().sort());
        
        // Convert to same dtype for meaningful comparison
        input = input.to(boundaries.scalar_type());
        
        // Extract parameters for bucketize
        bool out_int32 = false;
        bool right = false;
        
        // Use remaining bytes to determine parameters if available
        if (offset + 1 < Size) {
            out_int32 = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            right = Data[offset++] & 0x1;
        }
        
        // Apply bucketize operation
        torch::Tensor result = torch::bucketize(input, boundaries, out_int32, right);
        
        // Try different variants of the API
        try {
            // Try the out variant if we have more data
            if (offset < Size) {
                auto out_dtype = out_int32 ? torch::kInt32 : torch::kInt64;
                torch::Tensor output = torch::empty(input.sizes(), torch::TensorOptions().dtype(out_dtype));
                torch::bucketize_out(output, input, boundaries, out_int32, right);
            }
        } catch (const std::exception &) {
            // Silently ignore out variant failures (shape mismatches, etc.)
        }
        
        // Try with different parameters if we have more data
        if (offset < Size) {
            bool new_right = Data[offset++] & 0x1;
            torch::Tensor result2 = torch::bucketize(input, boundaries, out_int32, new_right);
        }
        
        // Try with different out_int32 parameter
        if (offset < Size) {
            bool new_out_int32 = Data[offset++] & 0x1;
            torch::Tensor result3 = torch::bucketize(input, boundaries, new_out_int32, right);
        }
        
        // Try scalar input variant
        if (offset + sizeof(float) <= Size) {
            float scalar_val;
            memcpy(&scalar_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Handle NaN/Inf
            if (std::isfinite(scalar_val)) {
                torch::Scalar scalar(scalar_val);
                torch::Tensor scalar_result = torch::bucketize(scalar, boundaries, out_int32, right);
            }
        }
        
        // Try with contiguous boundaries (explicit)
        torch::Tensor boundaries_contig = boundaries.contiguous();
        torch::Tensor result4 = torch::bucketize(input, boundaries_contig, out_int32, right);
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}