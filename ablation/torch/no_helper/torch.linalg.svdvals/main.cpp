#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// Helper to consume bytes from the fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t size, size_t& offset, T& value) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 16) return 0;  // Need minimum bytes for basic parameters
    
    try {
        size_t offset = 0;
        
        // Consume parameters for tensor construction
        uint8_t dtype_choice = 0;
        uint8_t batch_dims = 0;
        uint8_t m = 0, n = 0;
        uint8_t use_out_tensor = 0;
        uint8_t driver_choice = 0;
        uint8_t device_choice = 0;
        
        if (!consumeBytes(data, size, offset, dtype_choice)) return 0;
        if (!consumeBytes(data, size, offset, batch_dims)) return 0;
        if (!consumeBytes(data, size, offset, m)) return 0;
        if (!consumeBytes(data, size, offset, n)) return 0;
        if (!consumeBytes(data, size, offset, use_out_tensor)) return 0;
        if (!consumeBytes(data, size, offset, driver_choice)) return 0;
        if (!consumeBytes(data, size, offset, device_choice)) return 0;
        
        // Limit dimensions to reasonable values
        batch_dims = batch_dims % 4;  // 0-3 batch dimensions
        m = (m % 32) + 1;  // 1-32 rows
        n = (n % 32) + 1;  // 1-32 columns
        
        // Build shape vector
        std::vector<int64_t> shape;
        for (uint8_t i = 0; i < batch_dims; ++i) {
            uint8_t batch_size = 0;
            if (!consumeBytes(data, size, offset, batch_size)) {
                batch_size = 2;  // Default batch size
            }
            shape.push_back((batch_size % 5) + 1);  // 1-5 per batch dimension
        }
        shape.push_back(m);
        shape.push_back(n);
        
        // Select dtype
        torch::ScalarType dtype;
        switch (dtype_choice % 4) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kComplexFloat; break;
            case 3: dtype = torch::kComplexDouble; break;
            default: dtype = torch::kFloat32;
        }
        
        // Select device
        torch::Device device(torch::kCPU);
        #ifdef USE_CUDA
        if (torch::cuda::is_available() && (device_choice % 2 == 1)) {
            device = torch::Device(torch::kCUDA);
        }
        #endif
        
        // Create input tensor with remaining fuzzer data
        torch::Tensor A;
        size_t num_elements = 1;
        for (auto dim : shape) {
            num_elements *= dim;
        }
        
        // Calculate bytes needed based on dtype
        size_t bytes_per_element = torch::elementSize(dtype);
        size_t total_bytes_needed = num_elements * bytes_per_element;
        
        if (offset + total_bytes_needed <= size) {
            // Use fuzzer data directly
            A = torch::from_blob(
                const_cast<uint8_t*>(data + offset),
                shape,
                torch::TensorOptions().dtype(dtype)
            ).clone().to(device);
        } else {
            // Generate random tensor if not enough data
            A = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device));
        }
        
        // Prepare optional parameters
        torch::optional<torch::Tensor> out_tensor = torch::nullopt;
        if (use_out_tensor % 2 == 1) {
            // Create output tensor with appropriate shape
            std::vector<int64_t> out_shape = shape;
            out_shape.pop_back();  // Remove n dimension
            out_shape[out_shape.size() - 1] = std::min(m, n);  // Last dim is min(m, n)
            
            // Output is always real-valued
            torch::ScalarType out_dtype = (dtype == torch::kComplexFloat || dtype == torch::kComplexDouble) 
                ? (dtype == torch::kComplexFloat ? torch::kFloat32 : torch::kFloat64)
                : dtype;
            
            out_tensor = torch::empty(out_shape, torch::TensorOptions().dtype(out_dtype).device(device));
        }
        
        // Select driver (only relevant for CUDA)
        const char* driver = nullptr;
        #ifdef USE_CUDA
        if (device.is_cuda()) {
            switch (driver_choice % 5) {
                case 0: driver = nullptr; break;
                case 1: driver = "gesvd"; break;
                case 2: driver = "gesvdj"; break;
                case 3: driver = "gesvda"; break;
                case 4: driver = nullptr; break;
            }
        }
        #endif
        
        // Call torch.linalg.svdvals
        torch::Tensor result;
        if (out_tensor.has_value()) {
            if (driver != nullptr) {
                // With driver and out tensor (CUDA only)
                result = torch::linalg::svdvals_out(
                    out_tensor.value(),
                    A,
                    torch::optional<c10::string_view>(driver)
                );
            } else {
                // With out tensor, no driver
                result = torch::linalg::svdvals_out(
                    out_tensor.value(),
                    A
                );
            }
        } else {
            if (driver != nullptr) {
                // With driver, no out tensor (CUDA only)
                result = torch::linalg::svdvals(
                    A,
                    torch::optional<c10::string_view>(driver)
                );
            } else {
                // No driver, no out tensor
                result = torch::linalg::svdvals(A);
            }
        }
        
        // Perform some basic operations to exercise the result
        if (result.numel() > 0) {
            auto sum = result.sum();
            auto max_val = result.max();
            auto min_val = result.min();
            
            // Check that singular values are in descending order
            if (result.dim() > 0 && result.size(-1) > 1) {
                auto diff = result.narrow(-1, 1, result.size(-1) - 1) - 
                           result.narrow(-1, 0, result.size(-1) - 1);
                auto all_descending = (diff <= 0).all();
            }
        }
        
    } catch (const c10::Error& e) {
        // PyTorch-specific errors
        return 0;  // Continue fuzzing
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;  // Discard input for unexpected errors
    } catch (...) {
        // Unknown errors
        return -1;
    }
    
    return 0;
}