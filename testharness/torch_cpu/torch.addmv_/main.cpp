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
        
        // Need at least 3 tensors for addmv_: self, mat, vec
        if (Size < 6) // Minimum bytes needed for basic tensor creation
            return 0;
        
        // Create the self tensor (will be modified in-place)
        torch::Tensor self = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create the mat tensor (2D matrix)
        if (offset >= Size)
            return 0;
        torch::Tensor mat = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create the vec tensor (1D vector)
        if (offset >= Size)
            return 0;
        torch::Tensor vec = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get beta and alpha values if there's data left
        double beta = 1.0;
        double alpha = 1.0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&beta, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Create a copy of self for verification
        torch::Tensor self_copy = self.clone();
        
        // Apply the addmv_ operation (in-place)
        self.addmv_(mat, vec, beta, alpha);
        
        // Verify the operation with non-in-place version
        // This helps ensure the in-place operation behaves correctly
        torch::Tensor expected = self_copy.addmv(mat, vec, beta, alpha);
        
        // Check if the in-place operation produced the expected result
        if (self.defined() && expected.defined()) {
            try {
                bool equal = torch::allclose(self, expected);
                if (!equal) {
                    std::cerr << "In-place and out-of-place operations produced different results" << std::endl;
                }
            } catch (const std::exception& e) {
                // Comparison might fail for certain dtypes or shapes
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
