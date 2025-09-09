#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>
#include <numpy/arrayobject.h>
#include <Python.h>
#include <vector>
#include <cstring>

// Initialize Python and NumPy
class PythonInitializer {
public:
    PythonInitializer() {
        if (!Py_IsInitialized()) {
            Py_Initialize();
            import_array();
        }
    }
    ~PythonInitializer() {
        // Don't finalize Python in destructor as it may be used elsewhere
    }
};

static PythonInitializer python_init;

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 8) return 0; // Need at least basic parameters
        
        // Extract parameters from fuzzer input
        uint8_t dtype_choice = Data[offset++] % 8; // 8 different dtypes
        uint8_t ndim = (Data[offset++] % 4) + 1; // 1-4 dimensions
        uint8_t shape_seed = Data[offset++];
        uint8_t flags_seed = Data[offset++];
        uint32_t data_size_limit = 1024; // Limit array size for performance
        
        if (offset >= Size) return 0;
        
        // Determine NumPy dtype
        int numpy_dtype;
        size_t element_size;
        switch (dtype_choice) {
            case 0: numpy_dtype = NPY_FLOAT32; element_size = 4; break;
            case 1: numpy_dtype = NPY_FLOAT64; element_size = 8; break;
            case 2: numpy_dtype = NPY_INT32; element_size = 4; break;
            case 3: numpy_dtype = NPY_INT64; element_size = 8; break;
            case 4: numpy_dtype = NPY_INT16; element_size = 2; break;
            case 5: numpy_dtype = NPY_INT8; element_size = 1; break;
            case 6: numpy_dtype = NPY_UINT8; element_size = 1; break;
            case 7: numpy_dtype = NPY_BOOL; element_size = 1; break;
            default: numpy_dtype = NPY_FLOAT32; element_size = 4; break;
        }
        
        // Generate dimensions
        std::vector<npy_intp> dims(ndim);
        size_t total_elements = 1;
        for (int i = 0; i < ndim; i++) {
            if (offset >= Size) return 0;
            dims[i] = (Data[offset++] % 8) + 1; // 1-8 size per dimension
            total_elements *= dims[i];
            if (total_elements > data_size_limit / element_size) {
                dims[i] = 1; // Reduce size to stay within limits
                total_elements = 1;
                for (int j = 0; j <= i; j++) {
                    total_elements *= dims[j];
                }
            }
        }
        
        // Create NumPy array
        PyObject* numpy_array = PyArray_SimpleNew(ndim, dims.data(), numpy_dtype);
        if (!numpy_array) {
            return 0;
        }
        
        // Fill array with fuzzer data
        void* array_data = PyArray_DATA((PyArrayObject*)numpy_array);
        size_t array_bytes = total_elements * element_size;
        size_t available_bytes = Size - offset;
        
        if (available_bytes >= array_bytes) {
            std::memcpy(array_data, Data + offset, array_bytes);
            offset += array_bytes;
        } else {
            // Fill with pattern if not enough data
            for (size_t i = 0; i < array_bytes; i++) {
                ((uint8_t*)array_data)[i] = (offset + i < Size) ? Data[offset + i] : (uint8_t)(i % 256);
            }
        }
        
        // Test different array configurations
        PyArrayObject* arr = (PyArrayObject*)numpy_array;
        
        // Test 1: Basic from_numpy conversion
        torch::Tensor tensor1 = torch::from_numpy(arr);
        
        // Test 2: Test with different strides (if possible)
        if (ndim > 1 && PyArray_IS_C_CONTIGUOUS(arr)) {
            // Create a transposed view to test non-contiguous arrays
            PyObject* transposed = PyArray_Transpose(arr, nullptr);
            if (transposed) {
                torch::Tensor tensor2 = torch::from_numpy((PyArrayObject*)transposed);
                Py_DECREF(transposed);
            }
        }
        
        // Test 3: Test with sliced array (non-contiguous)
        if (total_elements > 2) {
            PyObject* slice = PySlice_New(nullptr, nullptr, PyLong_FromLong(2));
            if (slice) {
                PyObject* sliced = PyObject_GetItem(numpy_array, slice);
                if (sliced && PyArray_Check(sliced)) {
                    torch::Tensor tensor3 = torch::from_numpy((PyArrayObject*)sliced);
                    Py_DECREF(sliced);
                }
                Py_DECREF(slice);
            }
        }
        
        // Test 4: Verify tensor properties
        if (tensor1.defined()) {
            auto sizes = tensor1.sizes();
            auto strides = tensor1.strides();
            auto dtype = tensor1.dtype();
            auto device = tensor1.device();
            
            // Test basic operations on the tensor
            if (tensor1.numel() > 0) {
                auto sum = tensor1.sum();
                auto mean_val = tensor1.mean();
                auto reshaped = tensor1.reshape({-1});
            }
        }
        
        // Test 5: Edge cases with zero-sized dimensions
        if (offset + 1 < Size && Data[offset] % 10 == 0) {
            std::vector<npy_intp> zero_dims = {0, 5};
            PyObject* zero_array = PyArray_SimpleNew(2, zero_dims.data(), NPY_FLOAT32);
            if (zero_array) {
                torch::Tensor zero_tensor = torch::from_numpy((PyArrayObject*)zero_array);
                Py_DECREF(zero_array);
            }
        }
        
        // Test 6: Single element arrays
        if (offset + 1 < Size && Data[offset] % 7 == 0) {
            std::vector<npy_intp> single_dim = {1};
            PyObject* single_array = PyArray_SimpleNew(1, single_dim.data(), numpy_dtype);
            if (single_array) {
                torch::Tensor single_tensor = torch::from_numpy((PyArrayObject*)single_array);
                Py_DECREF(single_array);
            }
        }
        
        // Cleanup
        Py_DECREF(numpy_array);
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    catch (...)
    {
        std::cout << "Unknown exception caught" << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}