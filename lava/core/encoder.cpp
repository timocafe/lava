//
//  codec.cpp
//  experiment
//
//  Created by timothee.ewart on 5/6/20.
//

#include "ilc/include/encoder.h"
#include "ilc/utils/utils.h"

#include <filesystem>

namespace ilc {

encoder_helper::encoder_helper(const std::string &model_path) {
    control_tf_log();
    model_path_ = std::filesystem::path(model_path);
    const auto models = model_path_builder(model_path_);

    std::for_each(models.begin(), models.end(), [&](const auto &it) {
        const std::string model_path = it.first.native() + "/";
        const std::string model_name = it.second;
        encoder_.add_model(model_path, model_name);
    });
}

void encoder_helper::encode(const std::string &input, const std::string &output) {
    prepare_files(encoder_, input, output); // prepare input
    encoder_();                             // execute the TBB encoder
}

void encoder_helper::encode(const std::string &input_dir, const std::string &output_dir,
                            const std::string &model_name) {
    model(model_name);
    encode(input_dir, output_dir);
}

void encoder_helper::model(const std::string model) {
    std::filesystem::path path = model_path_ / model;
    if (!std::filesystem::is_directory(path))
        throw std::runtime_error("Invalid model.\n\n\t- Could not find " + path.string());
    // change model
    encoder_.model() = make_sha256(model_path_.string(), model);
}

void encoder_helper::ntokens(const std::size_t tokens) { encoder_.ntokens() = tokens; }

void encoder_helper::gpu(const std::size_t gpu) { encoder_.add_gpu_devices(gpu); }

void encoder_helper::tiling(const bool btiling) { encoder_.tiling() = btiling; }

void encoder_helper::canvas(const bool bcanvas) { encoder_.canvas() = bcanvas; }

} // namespace ilc
