#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic parameters
        if (Size < 16) return 0;

        // Extract matrix dimensions
        int rows = extractInt(Data, Size, offset, 1, 10);
        int cols = extractInt(Data, Size, offset, 1, 10);
        
        // matrix_exp requires square matrices
        int dim = std::min(rows, cols);
        
        // Extract data type
        int dtype_idx = extractInt(Data, Size, offset, 0, 3);
        torch::ScalarType dtype;
        switch (dtype_idx) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kComplexFloat; break;
            case 3: dtype = torch::kComplexDouble; break;
            default: dtype = torch::kFloat32; break;
        }

        // Extract device type
        torch::Device device = extractDevice(Data, Size, offset);

        // Create square matrix
        torch::Tensor A = torch::randn({dim, dim}, torch::TensorOptions().dtype(dtype).device(device));
        
        // Fill matrix with fuzzed data
        if (dtype == torch::kFloat32) {
            auto accessor = A.accessor<float, 2>();
            for (int i = 0; i < dim && offset < Size; i++) {
                for (int j = 0; j < dim && offset < Size; j++) {
                    accessor[i][j] = extractFloat(Data, Size, offset);
                }
            }
        } else if (dtype == torch::kFloat64) {
            auto accessor = A.accessor<double, 2>();
            for (int i = 0; i < dim && offset < Size; i++) {
                for (int j = 0; j < dim && offset < Size; j++) {
                    accessor[i][j] = extractDouble(Data, Size, offset);
                }
            }
        } else if (dtype == torch::kComplexFloat) {
            auto accessor = A.accessor<c10::complex<float>, 2>();
            for (int i = 0; i < dim && offset < Size; i++) {
                for (int j = 0; j < dim && offset < Size; j++) {
                    float real = extractFloat(Data, Size, offset);
                    float imag = extractFloat(Data, Size, offset);
                    accessor[i][j] = c10::complex<float>(real, imag);
                }
            }
        } else if (dtype == torch::kComplexDouble) {
            auto accessor = A.accessor<c10::complex<double>, 2>();
            for (int i = 0; i < dim && offset < Size; i++) {
                for (int j = 0; j < dim && offset < Size; j++) {
                    double real = extractDouble(Data, Size, offset);
                    double imag = extractDouble(Data, Size, offset);
                    accessor[i][j] = c10::complex<double>(real, imag);
                }
            }
        }

        // Test basic matrix_exp
        torch::Tensor result = torch::matrix_exp(A);
        
        // Verify result properties
        if (result.sizes() != A.sizes()) {
            std::cerr << "matrix_exp result has wrong shape" << std::endl;
        }
        
        if (result.dtype() != A.dtype()) {
            std::cerr << "matrix_exp result has wrong dtype" << std::endl;
        }

        // Test edge cases with special matrices
        if (offset < Size) {
            int special_case = extractInt(Data, Size, offset, 0, 4);
            
            switch (special_case) {
                case 0: {
                    // Zero matrix
                    torch::Tensor zero_mat = torch::zeros({dim, dim}, torch::TensorOptions().dtype(dtype).device(device));
                    torch::Tensor zero_exp = torch::matrix_exp(zero_mat);
                    break;
                }
                case 1: {
                    // Identity matrix
                    torch::Tensor eye_mat = torch::eye(dim, torch::TensorOptions().dtype(dtype).device(device));
                    torch::Tensor eye_exp = torch::matrix_exp(eye_mat);
                    break;
                }
                case 2: {
                    // Diagonal matrix
                    torch::Tensor diag_vals = torch::randn({dim}, torch::TensorOptions().dtype(dtype).device(device));
                    torch::Tensor diag_mat = torch::diag(diag_vals);
                    torch::Tensor diag_exp = torch::matrix_exp(diag_mat);
                    break;
                }
                case 3: {
                    // Upper triangular matrix
                    torch::Tensor upper_mat = torch::triu(A);
                    torch::Tensor upper_exp = torch::matrix_exp(upper_mat);
                    break;
                }
                case 4: {
                    // Lower triangular matrix
                    torch::Tensor lower_mat = torch::tril(A);
                    torch::Tensor lower_exp = torch::matrix_exp(lower_mat);
                    break;
                }
            }
        }

        // Test with different matrix sizes if we have more data
        if (offset < Size) {
            int new_dim = extractInt(Data, Size, offset, 1, 5);
            torch::Tensor small_mat = torch::randn({new_dim, new_dim}, torch::TensorOptions().dtype(dtype).device(device));
            torch::Tensor small_exp = torch::matrix_exp(small_mat);
        }

        // Test batched matrices
        if (offset < Size && dim <= 5) {
            int batch_size = extractInt(Data, Size, offset, 1, 3);
            torch::Tensor batch_mat = torch::randn({batch_size, dim, dim}, torch::TensorOptions().dtype(dtype).device(device));
            torch::Tensor batch_exp = torch::matrix_exp(batch_mat);
            
            if (batch_exp.sizes() != batch_mat.sizes()) {
                std::cerr << "Batched matrix_exp result has wrong shape" << std::endl;
            }
        }

        // Force evaluation of results
        result.sum().item<double>();

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}