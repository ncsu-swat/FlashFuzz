#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        if (Size < 16) {
            return 0; // Need minimum bytes for basic parameters
        }

        size_t offset = 0;
        
        // Read basic parameters
        uint8_t rank = Data[offset++] % 5 + 1; // 1-5 dimensions
        uint8_t batch_dims = Data[offset++] % 3; // 0-2 batch dimensions
        uint8_t m = (Data[offset++] % 10) + 1; // rows: 1-10
        uint8_t n = (Data[offset++] % 10) + 1; // cols: 1-10
        uint8_t dtype_choice = Data[offset++] % 3; // float32, float64, complex64
        bool unpack_data = Data[offset++] & 1;
        bool unpack_pivots = Data[offset++] & 1;
        bool use_out = Data[offset++] & 1;
        
        // Build shape for LU_data tensor
        std::vector<int64_t> shape;
        for (int i = 0; i < batch_dims; i++) {
            shape.push_back((Data[offset++] % 3) + 1); // batch size 1-3
            if (offset >= Size) break;
        }
        shape.push_back(m);
        shape.push_back(n);
        
        // Build shape for pivots tensor (same batch dims, min(m,n) for last dim)
        std::vector<int64_t> pivot_shape;
        for (int i = 0; i < batch_dims; i++) {
            pivot_shape.push_back(shape[i]);
        }
        pivot_shape.push_back(std::min(m, n));
        
        // Select dtype
        torch::ScalarType dtype = torch::kFloat32;
        if (dtype_choice == 1) dtype = torch::kFloat64;
        else if (dtype_choice == 2) dtype = torch::kComplexFloat;
        
        // Create options
        auto options = torch::TensorOptions().dtype(dtype);
        auto pivot_options = torch::TensorOptions().dtype(torch::kInt32);
        
        // Create LU_data tensor with fuzzer data
        torch::Tensor LU_data;
        size_t total_elements = 1;
        for (auto s : shape) total_elements *= s;
        
        if (dtype == torch::kComplexFloat) {
            std::vector<c10::complex<float>> data_vec;
            for (size_t i = 0; i < total_elements; i++) {
                float real = 0.0f, imag = 0.0f;
                if (offset + 7 < Size) {
                    memcpy(&real, Data + offset, 4);
                    offset += 4;
                    memcpy(&imag, Data + offset, 4);
                    offset += 4;
                } else {
                    real = (offset < Size) ? static_cast<float>(Data[offset++]) / 128.0f : 0.0f;
                    imag = (offset < Size) ? static_cast<float>(Data[offset++]) / 128.0f : 0.0f;
                }
                data_vec.push_back(c10::complex<float>(real, imag));
            }
            LU_data = torch::from_blob(data_vec.data(), shape, options).clone();
        } else if (dtype == torch::kFloat64) {
            std::vector<double> data_vec;
            for (size_t i = 0; i < total_elements; i++) {
                double val = 0.0;
                if (offset + 7 < Size) {
                    memcpy(&val, Data + offset, 8);
                    offset += 8;
                } else {
                    val = (offset < Size) ? static_cast<double>(Data[offset++]) / 128.0 : 0.0;
                }
                data_vec.push_back(val);
            }
            LU_data = torch::from_blob(data_vec.data(), shape, options).clone();
        } else {
            std::vector<float> data_vec;
            for (size_t i = 0; i < total_elements; i++) {
                float val = 0.0f;
                if (offset + 3 < Size) {
                    memcpy(&val, Data + offset, 4);
                    offset += 4;
                } else {
                    val = (offset < Size) ? static_cast<float>(Data[offset++]) / 128.0f : 0.0f;
                }
                data_vec.push_back(val);
            }
            LU_data = torch::from_blob(data_vec.data(), shape, options).clone();
        }
        
        // Create LU_pivots tensor
        size_t pivot_elements = 1;
        for (auto s : pivot_shape) pivot_elements *= s;
        std::vector<int32_t> pivot_vec;
        for (size_t i = 0; i < pivot_elements; i++) {
            int32_t val = (offset < Size) ? static_cast<int32_t>(Data[offset++] % std::min(m, n)) : 0;
            pivot_vec.push_back(val);
        }
        torch::Tensor LU_pivots = torch::from_blob(pivot_vec.data(), pivot_shape, pivot_options).clone();
        
        // Test with different stride patterns
        if (offset < Size && Data[offset++] & 1) {
            LU_data = LU_data.transpose(-2, -1).contiguous().transpose(-2, -1);
        }
        
        // Call lu_unpack
        if (use_out && offset < Size && Data[offset++] & 1) {
            // Create output tensors
            std::vector<int64_t> p_shape = shape;
            p_shape[p_shape.size() - 2] = m;
            p_shape[p_shape.size() - 1] = m;
            
            std::vector<int64_t> l_shape = shape;
            l_shape[l_shape.size() - 2] = m;
            l_shape[l_shape.size() - 1] = std::min(m, n);
            
            std::vector<int64_t> u_shape = shape;
            u_shape[u_shape.size() - 2] = std::min(m, n);
            u_shape[u_shape.size() - 1] = n;
            
            torch::Tensor P_out = torch::empty(p_shape, options);
            torch::Tensor L_out = torch::empty(l_shape, options);
            torch::Tensor U_out = torch::empty(u_shape, options);
            
            auto result = torch::lu_unpack_out(P_out, L_out, U_out, LU_data, LU_pivots, unpack_data, unpack_pivots);
            
            // Access results to ensure computation
            if (unpack_data) {
                auto L = std::get<1>(result);
                auto U = std::get<2>(result);
                L.sum();
                U.sum();
            }
            if (unpack_pivots) {
                auto P = std::get<0>(result);
                P.sum();
            }
        } else {
            auto result = torch::lu_unpack(LU_data, LU_pivots, unpack_data, unpack_pivots);
            
            // Access results to ensure computation
            if (unpack_data) {
                auto L = std::get<1>(result);
                auto U = std::get<2>(result);
                L.sum();
                U.sum();
            }
            if (unpack_pivots) {
                auto P = std::get<0>(result);
                P.sum();
            }
        }
        
        // Try edge cases
        if (offset < Size && Data[offset++] & 1) {
            // Empty tensor case
            torch::Tensor empty_lu = torch::empty({0, 0}, options);
            torch::Tensor empty_pivots = torch::empty({0}, pivot_options);
            try {
                torch::lu_unpack(empty_lu, empty_pivots, true, true);
            } catch (...) {
                // Expected to fail, continue
            }
        }
        
        if (offset < Size && Data[offset++] & 1) {
            // Single element case
            torch::Tensor single_lu = torch::randn({1, 1}, options);
            torch::Tensor single_pivots = torch::zeros({1}, pivot_options);
            torch::lu_unpack(single_lu, single_pivots, unpack_data, unpack_pivots);
        }
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}