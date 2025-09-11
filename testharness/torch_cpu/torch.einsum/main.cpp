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
        
        // Need at least 1 byte for equation selection and 1 byte for number of tensors
        if (Size < 2) {
            return 0;
        }
        
        // Parse number of tensors (1-3)
        uint8_t num_tensors = (Data[offset++] % 3) + 1;
        
        // Parse equation type
        uint8_t equation_selector = Data[offset++];
        
        // Create a vector to store our tensors
        std::vector<torch::Tensor> tensors;
        
        // Create tensors based on the parsed number
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                tensors.push_back(tensor);
            } catch (const std::exception& e) {
                // If tensor creation fails, just break the loop
                break;
            }
        }
        
        // Need at least one tensor to proceed
        if (tensors.empty()) {
            return 0;
        }
        
        // Define a set of einsum equations based on the number of tensors
        std::vector<std::string> equations_1tensor = {
            "i->i",           // Identity
            "i->",            // Sum all elements
            "...->...",       // Identity with ellipsis
            "ii->i",          // Diagonal
            "ii->",           // Trace
            "ij->ji",         // Transpose
            "ij->j",          // Row sum
            "ij->i",          // Column sum
            "ij->",           // Sum all elements
            "...i->...",      // Sum over last dimension
            "i...->...",      // Sum over first dimension
            "ijk->jki",       // Permute dimensions
            "ijk->",          // Sum all elements
            "ijkl->lkji",     // Reverse dimensions
            ""                // Empty equation (should be handled gracefully)
        };
        
        std::vector<std::string> equations_2tensor = {
            "i,i->i",         // Element-wise multiplication
            "i,i->",          // Dot product
            "i,j->ij",        // Outer product
            "ij,jk->ik",      // Matrix multiplication
            "ij,ij->ij",      // Element-wise multiplication
            "ij,ij->",        // Sum of element-wise products
            "ij,ji->",        // Trace of matrix product
            "ij,kl->ijkl",    // Tensor product
            "...i,...i->...", // Batch dot product
            "i...,...i->...", // Batch dot product with ellipsis
            "ij...,jk...->ik...", // Batch matrix multiplication
            "i,->i",          // Scalar multiplication
            ",i->i",          // Scalar multiplication
            "i,j,k->ijk",     // Invalid for 2 tensors
            ""                // Empty equation
        };
        
        std::vector<std::string> equations_3tensor = {
            "i,i,i->i",       // Element-wise multiplication of 3 tensors
            "i,j,k->ijk",     // Triple outer product
            "ij,jk,kl->il",   // Chain matrix multiplication
            "ij,jk,kl->ijkl", // Complex tensor contraction
            "i,i,i->",        // Triple dot product
            "ij,jk,ki->",     // Trace of triple product
            "...i,...j,...k->...ijk", // Batch outer product
            "i...,j...,k...->ijk...", // Batch outer product with ellipsis
            "i,j,->ij",       // Invalid for 3 tensors
            ""                // Empty equation
        };
        
        // Select the appropriate equation set based on number of tensors
        std::vector<std::string>* equation_set;
        if (num_tensors == 1) {
            equation_set = &equations_1tensor;
        } else if (num_tensors == 2) {
            equation_set = &equations_2tensor;
        } else {
            equation_set = &equations_3tensor;
        }
        
        // Select an equation from the set
        std::string equation = (*equation_set)[equation_selector % equation_set->size()];
        
        // Apply einsum operation
        try {
            torch::Tensor result;
            if (num_tensors == 1) {
                result = torch::einsum(equation, {tensors[0]});
            } else if (num_tensors == 2) {
                result = torch::einsum(equation, {tensors[0], tensors[1]});
            } else {
                result = torch::einsum(equation, {tensors[0], tensors[1], tensors[2]});
            }
            
            // Force evaluation of the result
            result.sizes();
        } catch (const std::exception& e) {
            // Expected exceptions from invalid inputs are fine
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
