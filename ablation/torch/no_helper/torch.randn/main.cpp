#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some bytes for basic fuzzing
        if (Size < 8) {
            return 0;
        }

        // Extract number of dimensions (1-6 to avoid excessive memory usage)
        uint8_t num_dims = (Data[offset++] % 6) + 1;
        
        // Extract dimensions
        std::vector<int64_t> sizes;
        for (int i = 0; i < num_dims && offset < Size; i++) {
            // Limit dimension size to avoid OOM (max 100 per dimension)
            int64_t dim_size = (Data[offset++] % 100) + 1;
            sizes.push_back(dim_size);
        }

        if (offset >= Size) {
            return 0;
        }

        // Extract dtype choice
        uint8_t dtype_choice = Data[offset++] % 4;
        torch::ScalarType dtype;
        switch (dtype_choice) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kFloat16; break;
            default: dtype = torch::kBFloat16; break;
        }

        // Extract device choice
        uint8_t device_choice = Data[offset++] % 2;
        torch::Device device = (device_choice == 0) ? torch::kCPU : torch::kCUDA;

        // Extract requires_grad flag
        bool requires_grad = (Data[offset++] % 2) == 1;

        // Extract pin_memory flag (only meaningful for CPU)
        bool pin_memory = (Data[offset++] % 2) == 1;

        // Test 1: Basic randn with size vector
        torch::TensorOptions options = torch::TensorOptions()
            .dtype(dtype)
            .device(device)
            .requires_grad(requires_grad)
            .pinned_memory(pin_memory && device.is_cpu());

        auto tensor1 = torch::randn(sizes, options);
        
        // Verify tensor properties
        if (tensor1.sizes().vec() != sizes) {
            throw std::runtime_error("Size mismatch in randn");
        }
        if (tensor1.dtype() != dtype) {
            throw std::runtime_error("Dtype mismatch in randn");
        }
        if (tensor1.device() != device) {
            throw std::runtime_error("Device mismatch in randn");
        }
        if (tensor1.requires_grad() != requires_grad) {
            throw std::runtime_error("requires_grad mismatch in randn");
        }

        // Test 2: randn with individual size arguments (up to 4D)
        if (sizes.size() >= 1) {
            auto tensor2 = torch::randn({sizes[0]}, options);
            if (tensor2.dim() != 1 || tensor2.size(0) != sizes[0]) {
                throw std::runtime_error("1D randn size mismatch");
            }
        }

        if (sizes.size() >= 2) {
            auto tensor3 = torch::randn({sizes[0], sizes[1]}, options);
            if (tensor3.dim() != 2 || tensor3.size(0) != sizes[0] || tensor3.size(1) != sizes[1]) {
                throw std::runtime_error("2D randn size mismatch");
            }
        }

        if (sizes.size() >= 3) {
            auto tensor4 = torch::randn({sizes[0], sizes[1], sizes[2]}, options);
            if (tensor4.dim() != 3) {
                throw std::runtime_error("3D randn dimension mismatch");
            }
        }

        // Test 3: randn with generator (if we have enough data)
        if (offset < Size) {
            uint64_t seed = 0;
            size_t seed_bytes = std::min(sizeof(seed), Size - offset);
            std::memcpy(&seed, Data + offset, seed_bytes);
            offset += seed_bytes;

            auto generator = torch::make_generator<torch::CPUGeneratorImpl>(seed);
            auto tensor_with_gen = torch::randn(sizes, generator, options);
            
            if (tensor_with_gen.sizes().vec() != sizes) {
                throw std::runtime_error("Size mismatch in randn with generator");
            }
        }

        // Test 4: randn with out parameter
        auto out_tensor = torch::empty(sizes, options);
        torch::randn_out(out_tensor, sizes);
        
        if (out_tensor.sizes().vec() != sizes) {
            throw std::runtime_error("Size mismatch in randn_out");
        }

        // Test 5: Edge cases
        // Empty tensor
        if (offset < Size && (Data[offset++] % 10) == 0) {
            auto empty_tensor = torch::randn({0}, options);
            if (empty_tensor.numel() != 0) {
                throw std::runtime_error("Empty tensor should have 0 elements");
            }
        }

        // Single element tensor
        if (offset < Size && (Data[offset++] % 10) == 1) {
            auto single_tensor = torch::randn({1}, options);
            if (single_tensor.numel() != 1) {
                throw std::runtime_error("Single tensor should have 1 element");
            }
        }

        // Test 6: Verify tensor contains finite values (not NaN/Inf)
        if (tensor1.numel() > 0 && tensor1.device().is_cpu()) {
            auto flat = tensor1.flatten();
            for (int64_t i = 0; i < std::min(flat.numel(), int64_t(10)); i++) {
                auto val = flat[i];
                if (dtype == torch::kFloat32) {
                    float f_val = val.item<float>();
                    if (!std::isfinite(f_val)) {
                        throw std::runtime_error("randn produced non-finite value");
                    }
                } else if (dtype == torch::kFloat64) {
                    double d_val = val.item<double>();
                    if (!std::isfinite(d_val)) {
                        throw std::runtime_error("randn produced non-finite value");
                    }
                }
            }
        }

        // Test 7: Multiple calls should produce different results (probabilistically)
        if (sizes.size() == 1 && sizes[0] == 1) {
            auto tensor_a = torch::randn({1}, options);
            auto tensor_b = torch::randn({1}, options);
            // Note: This is probabilistic, but extremely unlikely to be equal
            // We'll just ensure both are finite if on CPU
            if (tensor_a.device().is_cpu() && tensor_b.device().is_cpu()) {
                if (dtype == torch::kFloat32) {
                    float a_val = tensor_a.item<float>();
                    float b_val = tensor_b.item<float>();
                    if (!std::isfinite(a_val) || !std::isfinite(b_val)) {
                        throw std::runtime_error("randn produced non-finite values in comparison test");
                    }
                }
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}