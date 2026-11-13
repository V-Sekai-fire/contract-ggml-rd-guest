struct Slot {
  uint base;
  uint pad0;
  uint pad1;
  uint pad2;
};

[[vk::binding(0, 0)]]
StructuredBuffer<uint> params;
[[vk::binding(1, 0)]]
RWStructuredBuffer<float> s0;
[[vk::binding(2, 0)]]
RWStructuredBuffer<float> s1;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> s2;
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
  if ((e >= pw(53u))) {
    return;
  }
  uint nc = pw(19u);
  uint nr = ((pw(20u) * pw(21u)) * pw(22u));
  uint b1 = (pw(27u) + (e * pw(24u)));
  uint b2 = (pw(36u) + (e * pw(33u)));
  uint bd = (pw(9u) + (e * pw(6u)));
  float dn = (s0[pw(18u)] / float(nr));
  float mx = asfloat((4286578688u | (pw(0u) & 0u)));
  uint i = 0u;
  while ((i < nc)) {
    mx = max(mx, s1[(b1 + i)]);
    i = (i + 1u);
  }
  float se = 0.0f;
  i = 0u;
  while ((i < nc)) {
    se = (se + exp((s1[(b1 + i)] - mx)));
    i = (i + 1u);
  }
  float inv = (1.0f / se);
  i = 0u;
  while ((i < nc)) {
    dst[(bd + i)] = (((exp((s1[(b1 + i)] - mx)) * inv) - s2[(b2 + i)]) * dn);
    i = (i + 1u);
  }
}
