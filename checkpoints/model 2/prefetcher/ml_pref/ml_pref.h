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

  // ---- knobs ----
  static constexpr int   K_FEAT  = 16;    // feature dimension
  static constexpr int   MAX_OUT = 1;     // max prefetches per access
  static constexpr int   TIMEOUT = 1024;  // age (accesses) before marking "useless"
  static constexpr float ETA     = 0.05f; // learning rate
  static constexpr float L2      = 1e-4f; // L2 regularization

  // bootstrap & exploration
  static constexpr uint64_t BOOTSTRAP_CALLS = 50000; // kickstart learning
  static constexpr float    EPS_START = 0.10f;       // ε-greedy start
  static constexpr float    EPS_END   = 0.02f;       // ε-greedy floor
  static constexpr uint64_t EPS_DECAY = 200000;      // linear decay window

  // candidate distances in **lines** (not bytes)
  static constexpr std::array<int, 4> DISTS = { +1, +2, -1, -2 };

  // one logistic model per candidate distance: weights[action][feature]
  std::array<std::array<float, K_FEAT>, DISTS.size()> W{};

  struct Pending {
    int action;                                // which stride index
    std::array<float, K_FEAT> x;               // features at issue time
    int age;                                   // age in accesses
  };

  // pending prefetches keyed by a stable, line-aligned string form of the address
  std::unordered_map<std::string, Pending> pending;

  // tiny PC→last line & last stride memories
  std::unordered_map<std::string, champsim::block_number> pc_last_line; // prev line (typed)
  std::unordered_map<std::string, int>                    pc_last_stride; // -1 / 0 / +1

  // helpers
  static inline float sigmoid(float z) { return 1.0f / (1.0f + std::exp(-z)); }
  static std::string line_key(champsim::address a);  // defined in .cc
  static std::string to_key(champsim::address a);    // string key for IP map (in .cc)

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
