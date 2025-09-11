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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create src tensor (to be scattered into input)
        torch::Tensor src;
        if (offset < Size) {
            src = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we've consumed all data, create a simple tensor
            src = torch::ones_like(input);
        }
        
        // Get dim and index parameters for select_scatter
        int64_t dim = 0;
        int64_t index = 0;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&index, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply select_scatter operation
        // select_scatter(input, src, dim, index) scatters src into the input tensor
        // along the given dimension at the given index
        try {
            torch::Tensor result = torch::select_scatter(input, src, dim, index);
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected and part of testing
            // We don't want to discard these inputs
        }
        
        // Try with negative dimensions (valid in PyTorch for indexing from the end)
        try {
            if (input.dim() > 0) {
                int64_t neg_dim = -1;
                torch::Tensor result = torch::select_scatter(input, src, neg_dim, index);
            }
        } catch (const c10::Error& e) {
            // Expected for invalid inputs
        }
        
        // Try with negative indices
        try {
            torch::Tensor result = torch::select_scatter(input, src, dim, -1);
        } catch (const c10::Error& e) {
            // Expected for invalid inputs
        }
        
        // Try with extreme indices
        try {
            int64_t extreme_index = std::numeric_limits<int64_t>::max();
            torch::Tensor result = torch::select_scatter(input, src, dim, extreme_index);
        } catch (const c10::Error& e) {
            // Expected for invalid inputs
        }
        
        // Try with out-of-bounds dimensions
        try {
            int64_t out_of_bounds_dim = input.dim() + 10;
            torch::Tensor result = torch::select_scatter(input, src, out_of_bounds_dim, index);
        } catch (const c10::Error& e) {
            // Expected for invalid inputs
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
