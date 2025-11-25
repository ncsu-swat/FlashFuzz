#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/io_ops.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
}
}

template <typename T>
void fillTensorWithData(tensorflow::Tensor& tensor, const uint8_t* data,
                        size_t& offset, size_t total_size) {
    auto flat = tensor.flat<T>();
    const size_t num_elements = flat.size();
    const size_t element_size = sizeof(T);

    for (size_t i = 0; i < num_elements; ++i) {
        if (offset + element_size <= total_size) {
            T value;
            std::memcpy(&value, data + offset, element_size);
            offset += element_size;
            flat(i) = value;
        } else {
            flat(i) = T{};
        }
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) return 0;

    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        int64_t num_rows = static_cast<int64_t>((data[offset++] % 4) + 1);
        int64_t num_cols = static_cast<int64_t>((data[offset++] % 4) + 1);
        int64_t max_rows_in_memory = num_rows;

        const std::string ckpt_path = "/tmp/load_and_remap_matrix_fuzz";
        const std::string old_tensor_name = "matrix";

        tensorflow::Tensor ckpt_path_tensor(tensorflow::DT_STRING, {});
        ckpt_path_tensor.scalar<tensorflow::tstring>()() = ckpt_path;

        tensorflow::Tensor old_tensor_name_tensor(tensorflow::DT_STRING, {});
        old_tensor_name_tensor.scalar<tensorflow::tstring>()() = old_tensor_name;

        tensorflow::Tensor row_remapping_tensor(tensorflow::DT_INT64, {num_rows});
        auto row_vec = row_remapping_tensor.vec<int64_t>();
        for (int i = 0; i < num_rows; ++i) {
            row_vec(i) = i;
        }

        tensorflow::Tensor col_remapping_tensor(tensorflow::DT_INT64, {0});
        tensorflow::Tensor initializing_values_tensor(tensorflow::DT_FLOAT, {0});

        tensorflow::Tensor checkpoint_tensor(tensorflow::DT_FLOAT,
                                             {num_rows, num_cols});
        fillTensorWithData<float>(checkpoint_tensor, data, offset, size);

        auto* env = tensorflow::Env::Default();
        env->DeleteFile(ckpt_path).IgnoreError();
        env->DeleteFile(ckpt_path + ".index").IgnoreError();
        env->DeleteFile(ckpt_path + ".data-00000-of-00001").IgnoreError();

        tensorflow::BundleWriter writer(env, ckpt_path);
        tensorflow::Status bundle_status =
            writer.Add(old_tensor_name, checkpoint_tensor);
        if (!bundle_status.ok()) return 0;
        bundle_status = writer.Finish();
        if (!bundle_status.ok()) return 0;

        auto ckpt_path_placeholder = tensorflow::ops::Placeholder(
            root, tensorflow::DT_STRING,
            tensorflow::ops::Placeholder::Attrs().Shape({}));
        auto old_tensor_name_placeholder = tensorflow::ops::Placeholder(
            root, tensorflow::DT_STRING,
            tensorflow::ops::Placeholder::Attrs().Shape({}));
        auto row_remapping_placeholder = tensorflow::ops::Placeholder(
            root, tensorflow::DT_INT64,
            tensorflow::ops::Placeholder::Attrs().Shape({num_rows}));
        auto col_remapping_placeholder = tensorflow::ops::Placeholder(
            root, tensorflow::DT_INT64,
            tensorflow::ops::Placeholder::Attrs().Shape({0}));
        auto initializing_values_placeholder = tensorflow::ops::Placeholder(
            root, tensorflow::DT_FLOAT,
            tensorflow::ops::Placeholder::Attrs().Shape({0}));

        tensorflow::Node* load_node = nullptr;
        tensorflow::Status builder_status =
            tensorflow::NodeBuilder(root.GetUniqueNameForOp("LoadAndRemapMatrix"),
                                    "LoadAndRemapMatrix")
                .Input(ckpt_path_placeholder.output.node())
                .Input(old_tensor_name_placeholder.output.node())
                .Input(row_remapping_placeholder.output.node())
                .Input(col_remapping_placeholder.output.node())
                .Input(initializing_values_placeholder.output.node())
                .Attr("num_rows", num_rows)
                .Attr("num_cols", num_cols)
                .Attr("max_rows_in_memory", max_rows_in_memory)
                .Finalize(root.graph(), &load_node);
        if (!builder_status.ok()) {
            return 0;
        }

        tensorflow::Output load_and_remap_matrix(load_node, 0);

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;

        tensorflow::Status status = session.Run({
            {ckpt_path_placeholder, ckpt_path_tensor},
            {old_tensor_name_placeholder, old_tensor_name_tensor},
            {row_remapping_placeholder, row_remapping_tensor},
            {col_remapping_placeholder, col_remapping_tensor},
            {initializing_values_placeholder, initializing_values_tensor}
        }, {load_and_remap_matrix}, &outputs);

        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
