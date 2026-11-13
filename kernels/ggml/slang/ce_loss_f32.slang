struct Slot {
  uint base;
  uint pad0;
  uint pad1;
  uint pad2;
};

groupshared float sh[256];

[[vk::binding(0, 0)]]
StructuredBuffer<uint> params;
[[vk::binding(1, 0)]]
RWStructuredBuffer<float> s0;
[[vk::binding(2, 0)]]
RWStructuredBuffer<float> s1;
[[vk::binding(3, 0)]]
RWStructuredBuffer<uint> s2;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> dst;
[[vk::binding(0, 1)]]
ConstantBuffer<Slot> slot;

uint pw(uint k) {
  return params[(slot.base + k)];
}

uint off4(uint w, uint i0, uint i1, uint i2, uint i3) {
  return ((((pw((w + 8u)) + (i0 * pw((w + 4u)))) + (i1 * pw((w + 5u)))) + (i2 * pw((w + 6u)))) + (i3 * pw((w + 7u))));
}

[shader("compute")] [numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint e = ((((gid.y * pw(54u)) + gid.x) * 256u) + lid.x);
  uint nc = pw(10u);
  uint nr = ((pw(11u) * pw(12u)) * pw(13u));
  uint o0 = pw(18u);
  uint o1 = pw(27u);
  uint k0 = pw(15u);
  uint k1 = pw(24u);
  uint t = e;
  float acc = 0.0f;
  uint r = t;
  while ((r < nr)) {
    uint b0 = (o0 + (r * k0));
    uint b1 = (o1 + (r * k1));
    float mx = asfloat((4286578688u | (pw(0u) & 0u)));
    uint i = 0u;
    while ((i < nc)) {
      mx = max(mx, s0[(b0 + i)]);
      i = (i + 1u);
    }
    float se = 0.0f;
    i = 0u;
    while ((i < nc)) {
      se = (se + exp((s0[(b0 + i)] - mx)));
      i = (i + 1u);
    }
    float lse = log(se);
    float rs = 0.0f;
    i = 0u;
    while ((i < nc)) {
      rs = (rs + (((s0[(b0 + i)] - mx) - lse) * s1[(b1 + i)]));
      i = (i + 1u);
    }
    acc = (acc + rs);
    r = (r + 256u);
  }
  sh[t] = acc;
  GroupMemoryBarrierWithGroupSync();
  if ((t < 128u)) {
    sh[t] = (sh[t] + sh[(t + 128u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 64u)) {
    sh[t] = (sh[t] + sh[(t + 64u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 32u)) {
    sh[t] = (sh[t] + sh[(t + 32u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 16u)) {
    sh[t] = (sh[t] + sh[(t + 16u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 8u)) {
    sh[t] = (sh[t] + sh[(t + 8u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 4u)) {
    sh[t] = (sh[t] + sh[(t + 4u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 2u)) {
    sh[t] = (sh[t] + sh[(t + 2u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  if ((t < 1u)) {
    sh[t] = (sh[t] + sh[(t + 1u)]);
  }
  GroupMemoryBarrierWithGroupSync();
  float total = sh[0u];
  if ((e == 0u)) {
    dst[pw(9u)] = (total * ((-1.0f) / float(nr)));
  }
}
