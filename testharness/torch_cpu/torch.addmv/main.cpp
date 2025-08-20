#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }

        // Create input tensor 'input' (matrix)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create 'vec' tensor (vector)
        torch::Tensor vec;
        if (offset < Size) {
            vec = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a simple vector
            vec = torch::ones({1});
        }
        
        // Create 'bias' tensor (same shape as result of mat*vec)
        torch::Tensor bias;
        if (offset < Size) {
            bias = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a simple bias
            bias = torch::ones({1});
        }

        // Try to make tensors compatible for addmv if possible
        try {
            // Ensure input is 2D (matrix)
            if (input.dim() != 2) {
                if (input.dim() == 0) {
                    input = input.reshape({1, 1});
                } else if (input.dim() == 1) {
                    input = input.reshape({1, input.size(0)});
                } else {
                    // For higher dimensions, flatten to 2D
                    int64_t rows = input.size(0);
                    int64_t cols = 1;
                    for (int i = 1; i < input.dim(); i++) {
                        cols *= input.size(i);
                    }
                    input = input.reshape({rows, cols});
                }
            }

            // Ensure vec is 1D and matches input's second dimension
            if (vec.dim() != 1) {
                if (vec.dim() == 0) {
                    vec = vec.reshape({1});
                } else {
                    // Flatten higher dimensions to 1D
                    int64_t size = 1;
                    for (int i = 0; i < vec.dim(); i++) {
                        size *= vec.size(i);
                    }
                    vec = vec.reshape({size});
                }
            }

            // Ensure bias is 1D and matches input's first dimension
            if (bias.dim() != 1) {
                if (bias.dim() == 0) {
                    bias = bias.reshape({1});
                } else {
                    // Flatten higher dimensions to 1D
                    int64_t size = 1;
                    for (int i = 0; i < bias.dim(); i++) {
                        size *= bias.size(i);
                    }
                    bias = bias.reshape({size});
                }
            }
        } catch (const std::exception& e) {
            // If reshaping fails, just continue with original tensors
        }

        // Try different alpha and beta values
        double alpha = 1.0;
        double beta = 1.0;
        
        // Use some bytes from the input to determine alpha and beta if available
        if (offset + 1 < Size) {
            alpha = static_cast<double>(Data[offset]) / 128.0;
            offset++;
        }
        
        if (offset < Size) {
            beta = static_cast<double>(Data[offset]) / 128.0;
            offset++;
        }

        // Try to apply addmv operation
        try {
            torch::Tensor result = torch::addmv(bias, input, vec, alpha, beta);
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected for incompatible shapes
        }

        // Try alternative calling styles
        try {
            torch::Tensor result = torch::addmv(bias, input, vec);
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected for incompatible shapes
        }

        try {
            torch::Tensor result = bias.addmv(input, vec, alpha, beta);
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected for incompatible shapes
        }

        try {
            torch::Tensor result = bias.addmv(input, vec);
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected for incompatible shapes
        }

        // Try with out parameter
        try {
            torch::Tensor out = torch::empty_like(bias);
            torch::addmv_out(out, bias, input, vec, alpha, beta);
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected for incompatible shapes
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}