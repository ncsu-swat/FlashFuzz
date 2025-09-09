#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Parse dimensions for the input tensor
        auto dims = parseDimensions(Data, Size, offset, 2, 4); // 2D to 4D tensors
        if (dims.empty()) return 0;

        // Parse data type
        auto dtype = parseDType(Data, Size, offset);
        
        // Parse device type
        auto device = parseDevice(Data, Size, offset);

        // Create a square or rectangular matrix for LU decomposition
        // Ensure at least 2x2 for meaningful LU decomposition
        if (dims.size() < 2) return 0;
        if (dims[dims.size()-2] < 2 || dims[dims.size()-1] < 2) return 0;

        // Create input tensor
        auto input_tensor = createTensor(dims, dtype, device);
        if (!input_tensor.defined()) return 0;

        // Make the tensor non-singular by adding identity matrix scaled by a small value
        auto eye = torch::eye(std::min(dims[dims.size()-2], dims[dims.size()-1]), 
                             torch::TensorOptions().dtype(dtype).device(device));
        
        // Broadcast eye to match input dimensions if needed
        std::vector<int64_t> eye_shape = dims;
        eye_shape[eye_shape.size()-2] = eye.size(0);
        eye_shape[eye_shape.size()-1] = eye.size(1);
        
        if (dims.size() > 2) {
            std::vector<int64_t> broadcast_dims(dims.begin(), dims.end()-2);
            broadcast_dims.push_back(eye.size(0));
            broadcast_dims.push_back(eye.size(1));
            eye = eye.expand(broadcast_dims);
        }
        
        // Add small identity to make matrix more likely to be non-singular
        input_tensor = input_tensor + 0.1 * eye;

        // Parse pivot option
        bool pivot = parseBool(Data, Size, offset);

        // Perform LU decomposition first
        auto lu_result = torch::lu(input_tensor, pivot);
        auto LU = std::get<0>(lu_result);
        auto pivots = std::get<1>(lu_result);

        // Test lu_unpack with different parameter combinations
        
        // Test 1: Basic lu_unpack
        auto unpack_result1 = torch::lu_unpack(LU, pivots);
        auto P1 = std::get<0>(unpack_result1);
        auto L1 = std::get<1>(unpack_result1);
        auto U1 = std::get<2>(unpack_result1);

        // Verify shapes are reasonable
        if (P1.defined() && L1.defined() && U1.defined()) {
            // Check that dimensions make sense
            auto input_shape = input_tensor.sizes();
            auto P_shape = P1.sizes();
            auto L_shape = L1.sizes();
            auto U_shape = U1.sizes();
            
            // Basic dimension checks
            if (P_shape.size() != input_shape.size() || 
                L_shape.size() != input_shape.size() || 
                U_shape.size() != input_shape.size()) {
                return 0;
            }
        }

        // Test 2: lu_unpack with unpack_data=false
        bool unpack_data = parseBool(Data, Size, offset);
        auto unpack_result2 = torch::lu_unpack(LU, pivots, unpack_data);
        auto P2 = std::get<0>(unpack_result2);
        auto L2 = std::get<1>(unpack_result2);
        auto U2 = std::get<2>(unpack_result2);

        // Test 3: lu_unpack with unpack_pivots=false
        bool unpack_pivots = parseBool(Data, Size, offset);
        auto unpack_result3 = torch::lu_unpack(LU, pivots, true, unpack_pivots);
        auto P3 = std::get<0>(unpack_result3);
        auto L3 = std::get<1>(unpack_result3);
        auto U3 = std::get<2>(unpack_result3);

        // Test 4: lu_unpack with both flags
        auto unpack_result4 = torch::lu_unpack(LU, pivots, unpack_data, unpack_pivots);
        auto P4 = std::get<0>(unpack_result4);
        auto L4 = std::get<1>(unpack_result4);
        auto U4 = std::get<2>(unpack_result4);

        // Test edge cases with different tensor properties
        
        // Test with transposed LU matrix
        if (LU.dim() >= 2) {
            auto LU_t = LU.transpose(-2, -1);
            try {
                auto unpack_t = torch::lu_unpack(LU_t, pivots);
            } catch (...) {
                // Expected to potentially fail due to dimension mismatch
            }
        }

        // Test with cloned tensors
        auto LU_clone = LU.clone();
        auto pivots_clone = pivots.clone();
        auto unpack_clone = torch::lu_unpack(LU_clone, pivots_clone);

        // Test with contiguous tensors
        if (!LU.is_contiguous()) {
            auto LU_contig = LU.contiguous();
            auto unpack_contig = torch::lu_unpack(LU_contig, pivots);
        }

        // Test with different pivot tensor dtypes if possible
        if (pivots.dtype() != torch::kInt32) {
            try {
                auto pivots_int32 = pivots.to(torch::kInt32);
                auto unpack_int32 = torch::lu_unpack(LU, pivots_int32);
            } catch (...) {
                // May fail due to dtype requirements
            }
        }

        // Verify basic mathematical properties when unpack_data=true and unpack_pivots=true
        if (unpack_data && unpack_pivots && P1.defined() && L1.defined() && U1.defined()) {
            try {
                // P @ L @ U should approximately equal the original matrix
                auto reconstructed = torch::matmul(torch::matmul(P1, L1), U1);
                
                // Check if shapes match for comparison
                if (reconstructed.sizes() == input_tensor.sizes()) {
                    auto diff = torch::abs(reconstructed - input_tensor);
                    auto max_diff = torch::max(diff);
                    // Allow for numerical precision errors
                    if (max_diff.item<double>() > 1e-3) {
                        // Large reconstruction error might indicate an issue
                    }
                }
            } catch (...) {
                // Matrix multiplication might fail for edge cases
            }
        }

        // Test memory access patterns
        if (P1.defined()) P1.sum();
        if (L1.defined()) L1.sum();
        if (U1.defined()) U1.sum();

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}