#ifndef PREFETCHER_ML_PREF_H
#define PREFETCHER_ML_PREF_H

#include <cstdint>
#include <array>
#include <unordered_map>
#include <string>
#include <cmath>

#include "address.h"
#include "modules.h"

struct ml_pref : public champsim::modules::prefetcher {
  using prefetcher::prefetcher;

  // ---- base knobs ----
  static constexpr int   K_FEAT      = 16;     // feature dimension
  static constexpr int   TIMEOUT     = 512;    // age (accesses) before marking "useless"
  static constexpr float ETA         = 0.05f;  // learning rate
  static constexpr float L2          = 1e-4f;  // L2 regularization

  // bootstrap & exploration
  static constexpr uint64_t BOOTSTRAP_CALLS = 30000;  // kickstart learning (+1)
  static constexpr float    EPS_START       = 0.20f;  // ε start
  static constexpr float    EPS_END         = 0.02f;  // ε floor
  static constexpr uint64_t EPS_DECAY       = 200000; // linear decay window

  // candidate distances in **lines**
  static constexpr std::array<int, 8> DISTS = { +1, +2, +4, +8, -1, -2, -4, -8 };

  // model: one logistic head per stride
  std::array<std::array<float, K_FEAT>, DISTS.size()> W{};

  struct Pending {
    int action;                                // stride index
    std::array<float, K_FEAT> x;               // features at issue time
    int age;                                   // age in accesses
  };

  // pending prefetches (keyed by line-aligned address string)
  std::unordered_map<std::string, Pending> pending;

  // tiny PC→last line & last stride memories (typed-safe)
  std::unordered_map<std::string, champsim::block_number> pc_last_line;
  std::unordered_map<std::string, int>                    pc_last_stride; // {-1,0,+1}

  // aggressiveness controller (runtime)
  int   max_out = 3;          // can rise to 3 when coverage is low
  float thr     = 0.45f;      // decision threshold (auto-tuned)

  // moving window stats (updated via useful/issued observed at this level)
  uint64_t win_issued = 0, win_useful = 0, win_demand_miss = 0;
  uint64_t win_calls  = 0;
  static constexpr uint64_t WIN_PERIOD = 4096;   // adjust every 4K calls

  // helpers
  static inline float sigmoid(float z) { return 1.0f / (1.0f + std::exp(-z)); }
  static std::string line_key(champsim::address a);  // in .cc
  static std::string to_key(champsim::address a);    // in .cc

  // feature builder & online update
  void features(std::array<float, K_FEAT>& f,
                champsim::address addr, champsim::address ip,
                uint8_t cache_hit, access_type type,
                int last_stride_for_ip) const;

  void update(int action, const std::array<float, K_FEAT>& x, float label);

  // required hooks
  uint32_t prefetcher_cache_operate(champsim::address addr,
                                    champsim::address ip,
                                    uint8_t cache_hit,
                                    bool useful_prefetch,
                                    access_type type,
                                    uint32_t metadata_in);

  uint32_t prefetcher_cache_fill(champsim::address addr,
                                 long set,
                                 long way,
                                 uint8_t prefetch,
                                 champsim::address evicted_addr,
                                 uint32_t metadata_in);
};

#endif // PREFETCHER_ML_PREF_H

