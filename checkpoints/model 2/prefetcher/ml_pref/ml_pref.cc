#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
#include <random>

#include "ml_pref.h"

// stable, line-aligned string key for an address
std::string ml_pref::line_key(champsim::address a) {
  std::ostringstream oss;
  champsim::block_number ln{a};      // line index
  oss << champsim::address{ln};      // operator<< for address_slice
  return oss.str();
}

// stable string from raw address (for IP map key)
std::string ml_pref::to_key(champsim::address a) {
  std::ostringstream oss; oss << a; return oss.str();
}

// handy string-hashers for address/ip when building buckets
static inline uint64_t hash_addr_to_u64(champsim::address a) {
  std::ostringstream oss; oss << a;
  return std::hash<std::string>{}(oss.str());
}
static inline uint64_t hash_ip_to_u64(champsim::address ip) {
  std::ostringstream oss; oss << ip;
  return std::hash<std::string>{}(oss.str());
}

// ---- features: bias, hashed IP bucket, page-offset bucket, hit flag, type,
//                and tiny PC-stride memory (three bits)
void ml_pref::features(std::array<float, K_FEAT>& f,
                       champsim::address addr, champsim::address ip,
                       uint8_t cache_hit, access_type type,
                       int last_stride_for_ip) const
{
  for (auto& v : f) v = 0.0f;
  f[0] = 1.0f; // bias

  // hashed IP bucket (4 buckets)
  uint64_t ipu = hash_ip_to_u64(ip);
  f[1 + (ipu & 0x3)] = 1.0f; // f[1..4]

  // "page offset quadrant": derive a pseudo-offset from hashed addr (portable)
  uint64_t au = hash_addr_to_u64(addr);
  uint32_t off = static_cast<uint32_t>(au & 0xFFFu);
  f[5 + ((off >> 10) & 0x3)] = 1.0f; // f[5..8]

  // last demand was hit?
  f[9] = cache_hit ? 1.0f : 0.0f;

  // access type one-hot
  if (type == access_type::LOAD)       f[10] = 1.0f;
  else if (type == access_type::WRITE) f[11] = 1.0f;
  else if (type == access_type::PREFETCH) f[12] = 1.0f;

  // tiny PC-stride features (only care about -1/0/+1)
  f[13] = (last_stride_for_ip == +1) ? 1.0f : 0.0f;
  f[14] = (last_stride_for_ip == -1) ? 1.0f : 0.0f;
  f[15] = (last_stride_for_ip ==  0) ? 1.0f : 0.0f;
}

// ---- online logistic update (per-action)
void ml_pref::update(int a, const std::array<float, K_FEAT>& x, float y)
{
  auto& w = W[a];
  float dot = 0.0f;
  for (int i = 0; i < K_FEAT; ++i) dot += w[i] * x[i];
  float yhat = sigmoid(dot);
  float g = (y - yhat);
  for (int i = 0; i < K_FEAT; ++i) {
    w[i] = w[i] * (1.0f - ETA * L2) + ETA * g * x[i];
  }
}

// ---- main hook
uint32_t ml_pref::prefetcher_cache_operate(champsim::address addr,
                                           champsim::address ip,
                                           uint8_t cache_hit,
                                           bool useful_prefetch,
                                           access_type type,
                                           uint32_t metadata_in)
{
  // +1 feedback: current demand used a prefetched line (at THIS cache)
  if (useful_prefetch) {
    auto it = pending.find(line_key(addr));
    if (it != pending.end()) {
      update(it->second.action, it->second.x, 1.0f);
      pending.erase(it);
    }
  }

  // timeouts -> 0 feedback
  if (!pending.empty()) {
    std::vector<std::string> dead;
    dead.reserve(pending.size());
    for (auto& kv : pending) {
      auto& p = kv.second;
      if (++p.age >= TIMEOUT) {
        update(p.action, p.x, 0.0f);
        dead.push_back(kv.first);
      }
    }
    for (auto& k : dead) pending.erase(k);
  }

  // ---- Bootstrap FIRST: force +1 for a while to kickstart learning
  static uint64_t calls = 0;
  if (calls++ < BOOTSTRAP_CALLS) {
    std::array<float, K_FEAT> x0{}; // zero features; we don't update on bootstrap itself
    champsim::address pf = champsim::address{champsim::block_number{addr} + 1};
    if (prefetch_line(pf, /*fill_this_level=*/true, /*metadata=*/metadata_in)) { // <<< FILL THIS LEVEL
      pending[line_key(pf)] = Pending{ /*action=*/0, x0, /*age=*/0 };
    }
    // fall through to update PC-stride memory
  }

  // tiny PC→last line & last stride memory update (per IP), without integer casts
  std::string ipk = to_key(ip);
  int last_stride = 0;
  auto it_prev = pc_last_line.find(ipk);
  champsim::block_number cur_line{addr};
  if (it_prev != pc_last_line.end()) {
    champsim::block_number prev_line = it_prev->second;
    if (cur_line == prev_line + 1)      last_stride = +1;
    else if (cur_line == prev_line - 1) last_stride = -1;
    else if (cur_line == prev_line)     last_stride =  0;
    else                                last_stride =  0;
    pc_last_stride[ipk] = last_stride;
  }
  pc_last_line[ipk] = cur_line;
  auto it_ls = pc_last_stride.find(ipk);
  if (it_ls != pc_last_stride.end()) last_stride = it_ls->second;

  // features for this context
  std::array<float, K_FEAT> x;
  features(x, addr, ip, cache_hit, type, last_stride);

  // score candidate strides
  struct Cand { int a; float score; champsim::address pf; };
  std::array<Cand, DISTS.size()> cand;
  champsim::block_number cur{addr};

  for (int i = 0; i < static_cast<int>(DISTS.size()); ++i) {
    // w·x
    float dot = 0.0f;
    for (int j = 0; j < K_FEAT; ++j) dot += W[i][j] * x[j];
    float p = sigmoid(dot);

    champsim::block_number pf_line = cur + DISTS[i];
    cand[i] = Cand{ i, p, champsim::address{pf_line} };
  }

  // sort top-1
  std::partial_sort(cand.begin(), cand.begin() + 1, cand.end(),
                    [](const Cand& a, const Cand& b){ return a.score > b.score; });

  // ε-greedy exploration (decays over EPS_DECAY calls)
  static std::minstd_rand rng(0xC0FFEE);
  std::uniform_real_distribution<float> uni(0.0f, 1.0f);
  float frac = (calls < EPS_DECAY) ? (float)calls / (float)EPS_DECAY : 1.0f;
  float eps = EPS_START + (EPS_END - EPS_START) * frac; // linear decay
  int choice = 0;
  if (uni(rng) < eps) {
    // explore: try +1 vs -1 whichever currently scores higher
    int a_pos = 0; // +1 index in DISTS
    int a_neg = 2; // -1 index in DISTS
    choice = (cand[a_pos].score >= cand[a_neg].score) ? a_pos : a_neg;
  } else {
    // exploit best
    choice = 0; // top-1 due to partial_sort
  }

  int issued = 0;
  // inclusive threshold 0.5, and FILL THIS LEVEL so usefulness is counted here
  if (cand[choice].score >= 0.5f) {
    if (prefetch_line(cand[choice].pf, /*fill_this_level=*/true, /*metadata=*/metadata_in)) { // <<< FILL THIS LEVEL
      pending[line_key(cand[choice].pf)] = Pending{ cand[choice].a, x, 0 };
      ++issued;
    }
  }

  // ---- Fail-safe: if we haven't issued for a while, force a +1 to re-prime learning
  static uint64_t since_issue = 0;
  if (issued == 0) {
    ++since_issue;
    if (since_issue >= 2048) { // 2K accesses idle → one nudge
      champsim::address pf = champsim::address{champsim::block_number{addr} + 1};
      if (prefetch_line(pf, /*fill_this_level=*/true, /*metadata=*/metadata_in)) { // <<< FILL THIS LEVEL
        pending[line_key(pf)] = Pending{ /*action=*/0, x, 0 };
      }
      since_issue = 0;
    }
  } else {
    since_issue = 0;
  }

  return metadata_in;
}

// We learn in operate() via useful_prefetch (+1) and timeouts (0).
uint32_t ml_pref::prefetcher_cache_fill(champsim::address /*addr*/,
                                        long /*set*/,
                                        long /*way*/,
                                        uint8_t /*prefetch*/,
                                        champsim::address /*evicted_addr*/,
                                        uint32_t metadata_in)
{
  return metadata_in;
}

