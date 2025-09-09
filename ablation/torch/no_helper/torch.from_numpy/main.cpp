#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>
#include <numpy/arrayobject.h>
#include <Python.h>
#include <vector>
#include <cstring>

// Initialize Python and NumPy
static bool python_initialized = false;

void init_python_numpy() {
    if (!python_initialized) {
        Py_Initialize();
        import_array();
        python_initialized = true;
    }
}

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        init_python_numpy();
        
        size_t offset = 0;
        
        // Need at least 8 bytes for basic parameters
        if (Size < 8) return 0;
        
        // Extract fuzzing parameters
        uint8_t dtype_choice = Data[offset++] % 11; // 11 supported dtypes
        uint8_t ndim = (Data[offset++] % 4) + 1; // 1-4 dimensions
        uint8_t readonly_flag = Data[offset++] % 2; // 0 or 1
        
        // Extract dimension sizes (limit to reasonable values)
        std::vector<npy_intp> dims(ndim);
        size_t total_elements = 1;
        for (int i = 0; i < ndim && offset < Size; i++) {
            dims[i] = (Data[offset++] % 10) + 1; // 1-10 elements per dimension
            total_elements *= dims[i];
            if (total_elements > 1000) { // Prevent excessive memory usage
                total_elements = 1000;
                dims[i] = 1000 / (total_elements / dims[i]);
                break;
            }
        }
        
        // Map dtype_choice to NumPy dtypes
        int numpy_dtype;
        size_t element_size;
        switch (dtype_choice) {
            case 0: numpy_dtype = NPY_FLOAT64; element_size = 8; break;
            case 1: numpy_dtype = NPY_FLOAT32; element_size = 4; break;
            case 2: numpy_dtype = NPY_FLOAT16; element_size = 2; break;
            case 3: numpy_dtype = NPY_COMPLEX64; element_size = 8; break;
            case 4: numpy_dtype = NPY_COMPLEX128; element_size = 16; break;
            case 5: numpy_dtype = NPY_INT64; element_size = 8; break;
            case 6: numpy_dtype = NPY_INT32; element_size = 4; break;
            case 7: numpy_dtype = NPY_INT16; element_size = 2; break;
            case 8: numpy_dtype = NPY_INT8; element_size = 1; break;
            case 9: numpy_dtype = NPY_UINT8; element_size = 1; break;
            case 10: numpy_dtype = NPY_BOOL; element_size = 1; break;
            default: numpy_dtype = NPY_FLOAT32; element_size = 4; break;
        }
        
        // Calculate required data size
        size_t required_data_size = total_elements * element_size;
        size_t remaining_size = Size - offset;
        
        // Create NumPy array
        PyObject* array = PyArray_SimpleNew(ndim, dims.data(), numpy_dtype);
        if (!array) {
            return 0;
        }
        
        // Fill array with fuzz data (cycle through available data if needed)
        void* array_data = PyArray_DATA((PyArrayObject*)array);
        if (remaining_size > 0) {
            for (size_t i = 0; i < required_data_size; i++) {
                ((uint8_t*)array_data)[i] = Data[offset + (i % remaining_size)];
            }
        } else {
            // Fill with zeros if no data left
            memset(array_data, 0, required_data_size);
        }
        
        // Make array read-only if flag is set (to test warning condition)
        if (readonly_flag) {
            PyArray_CLEARFLAGS((PyArrayObject*)array, NPY_ARRAY_WRITEABLE);
        }
        
        // Test torch::from_numpy
        torch::Tensor tensor = torch::from_numpy((PyArrayObject*)array);
        
        // Test basic tensor properties
        auto sizes = tensor.sizes();
        auto dtype = tensor.dtype();
        auto device = tensor.device();
        
        // Test tensor operations that should work
        if (tensor.numel() > 0) {
            // Test element access (read)
            if (tensor.dim() == 1 && tensor.size(0) > 0) {
                auto first_element = tensor[0];
            }
            
            // Test basic operations
            auto sum_result = tensor.sum();
            auto mean_result = tensor.mean();
            
            // Test cloning (should work)
            auto cloned = tensor.clone();
            
            // Test memory sharing verification
            if (!readonly_flag && tensor.numel() > 0) {
                // Modify tensor and verify it affects the numpy array
                // Only for writable arrays to avoid undefined behavior
                if (tensor.dtype() == torch::kFloat32) {
                    tensor.fill_(42.0f);
                } else if (tensor.dtype() == torch::kInt32) {
                    tensor.fill_(42);
                } else if (tensor.dtype() == torch::kBool) {
                    tensor.fill_(true);
                }
            }
        }
        
        // Test edge cases
        if (tensor.dim() > 1) {
            // Test reshaping (should fail since tensor is not resizable)
            try {
                tensor.resize_({tensor.numel()});
            } catch (...) {
                // Expected to fail
            }
        }
        
        // Test conversion back to different types
        if (tensor.dtype().isFloatingPoint()) {
            auto int_tensor = tensor.to(torch::kInt32);
            auto double_tensor = tensor.to(torch::kFloat64);
        }
        
        // Clean up
        Py_DECREF(array);
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    catch (...) {
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }
    return 0; // keep the input
}