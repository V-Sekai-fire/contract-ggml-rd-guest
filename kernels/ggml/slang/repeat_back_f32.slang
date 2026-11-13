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

void unravel4(uint e, uint w, out uint i0, out uint i1, out uint i2, out uint i3) {
  uint n0 = pw(w);
  uint n1 = pw((w + 1u));
  uint n2 = pw((w + 2u));
  i0 = (e % n0);
  uint r = (e / n0);
  i1 = (r % n1);
  r = (r / n1);
  i2 = (r % n2);
  i3 = (r / n2);
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
  uint k0;
  uint k1;
  uint k2;
  uint k3;
  unravel4(e, 1u, k0, k1, k2, k3);
  uint d = off4(1u, k0, k1, k2, k3);
  float acc = 0.0f;
  uint i3 = 0u;
  while ((i3 < (pw(13u) / pw(4u)))) {
    uint i2 = 0u;
    while ((i2 < (pw(12u) / pw(3u)))) {
      uint i1 = 0u;
      while ((i1 < (pw(11u) / pw(2u)))) {
        uint i0 = 0u;
        while ((i0 < (pw(10u) / pw(1u)))) {
          acc = (acc + s0[((((pw(18u) + (((i0 * pw(1u)) + k0) * pw(14u))) + (((i1 * pw(2u)) + k1) * pw(15u))) + (((i2 * pw(3u)) + k2) * pw(16u))) + (((i3 * pw(4u)) + k3) * pw(17u)))]);
          i0 = (i0 + 1u);
        }
        i1 = (i1 + 1u);
      }
      i2 = (i2 + 1u);
    }
    i3 = (i3 + 1u);
  }
  dst[d] = acc;
}
