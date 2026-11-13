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
RWStructuredBuffer<uint> s2;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> dst;
[[vk::binding(0, 1)]]
ConstantBuffer<Slot> slot;

uint pw(uint k) {
  return params[(slot.base + k)];
}

float pf(uint k) {
  return asfloat(pw(k));
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
  uint nc = pw(10u);
  float scale = pf(37u);
  uint b0 = (pw(18u) + (e * pw(15u)));
  uint b1 = (pw(27u) + (e * pw(24u)));
  uint bd = (pw(9u) + (e * pw(6u)));
  float dot = 0.0f;
  uint i = 0u;
  while ((i < nc)) {
    dot = (dot + (s1[(b1 + i)] * s0[(b0 + i)]));
    i = (i + 1u);
  }
  i = 0u;
  while ((i < nc)) {
    dst[(bd + i)] = (((s0[(b0 + i)] - dot) * s1[(b1 + i)]) * scale);
    i = (i + 1u);
  }
}
