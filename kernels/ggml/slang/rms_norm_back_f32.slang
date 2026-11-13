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
  uint n1 = pw(11u);
  uint n2 = pw(12u);
  float eps = pf(37u);
  uint i1 = (e % n1);
  uint i2 = ((e / n1) % n2);
  uint i3 = ((e / n1) / n2);
  uint b0 = (((pw(18u) + (i1 * pw(15u))) + (i2 * pw(16u))) + (i3 * pw(17u)));
  uint b1 = (((pw(27u) + (i1 * pw(24u))) + (i2 * pw(25u))) + (i3 * pw(26u)));
  uint bd = (((pw(9u) + (i1 * pw(6u))) + (i2 * pw(7u))) + (i3 * pw(8u)));
  float sxx = 0.0f;
  float sxdz = 0.0f;
  uint i = 0u;
  while ((i < nc)) {
    float xv = s1[(b1 + i)];
    sxx = (sxx + (xv * xv));
    sxdz = (sxdz + (xv * s0[(b0 + i)]));
    i = (i + 1u);
  }
  float rrms = (1.0f / sqrt(((sxx / float(nc)) + eps)));
  float sx = ((0.0f - sxdz) / (sxx + (eps * float(nc))));
  i = 0u;
  while ((i < nc)) {
    dst[(bd + i)] = ((s0[(b0 + i)] + (s1[(b1 + i)] * sx)) * rrms);
    i = (i + 1u);
  }
}
