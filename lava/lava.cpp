#include <filesystem>
#include <string>

#include "lava/distributed/decoder.h"
#include "lava/distributed/node.h"
#include "lava/path.h"
#include "lava/utils/utils.h"

#include "lava.h"

void generate(const char *s) {
  std::string name(s);
  const auto &model = lava::helper_build_path::model_path() + name;
  lava::lavadom l(model);
  l();
  std::cout << " fini ! \n";
}
