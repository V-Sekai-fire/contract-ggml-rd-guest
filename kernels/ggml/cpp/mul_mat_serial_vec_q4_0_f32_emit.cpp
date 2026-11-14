#include "/home/setup/tools/slang-2026.20/include/slang-cpp-prelude.h"

#ifdef SLANG_PRELUDE_NAMESPACE
using namespace SLANG_PRELUDE_NAMESPACE;
#endif


#line 1 "slang/mul_mat_serial_vec_q4_0_f32.slang"
struct Slot_0
{
    uint32_t base_0;
    uint32_t pad0_0;
    uint32_t pad1_0;
    uint32_t pad2_0;
};


#line 1
struct GlobalParams_0
{
    StructuredBuffer<uint32_t> params_0;
    RWStructuredBuffer<uint32_t> s0_0;
    RWStructuredBuffer<float> s1_0;
    RWStructuredBuffer<uint32_t> s2_0;
    RWStructuredBuffer<float> dst_0;
    Slot_0* slot_0;
};


#line 1
struct KernelContext_0
{
    GlobalParams_0* globalParams_0;
};


#line 21
static uint32_t pw_0(uint32_t k_0, KernelContext_0 * kernelContext_0)
{

#line 22
    return kernelContext_0->globalParams_0->params_0.Load(kernelContext_0->globalParams_0->slot_0->base_0 + k_0);
}

static float f16_bits_to_f32_0(uint32_t h_0)
{

#line 26
    uint32_t s_0 = (h_0 & 32768U) << 16U;
    uint32_t x_0 = h_0 & 31744U;
    uint32_t b_0 = ((h_0 & 32767U) << 13U) + 939524096U;

#line 28
    uint32_t b_1;
    if(x_0 == 31744U)
    {

#line 29
        b_1 = b_0 + 939524096U;

#line 29
    }
    else
    {

#line 29
        b_1 = b_0;

#line 29
    }


    if(x_0 == 0U)
    {

#line 32
        b_1 = (F32_asuint(((U32_asfloat((b_1))) - (U32_asfloat((947912704U))))));

#line 32
    }


    return (U32_asfloat((b_1 | s_0)));
}

static float lda_0(uint32_t e_0, KernelContext_0 * kernelContext_1)
{
    uint32_t w_0 = e_0 & 31U;
    uint32_t _S1 = (e_0 >> 5U) * 18U;

#line 41
    uint32_t _S2 = pw_0(53U, kernelContext_1);

#line 41
    uint32_t bo_0 = _S1 + _S2;
    uint32_t dh_0 = ((*(&(kernelContext_1->globalParams_0->s0_0)[bo_0 >> 2U])) >> ((bo_0 & 2U) * 8U)) & 65535U;
    uint32_t qo_0 = bo_0 + 2U + (w_0 & 15U);
    uint32_t qb_0 = ((*(&(kernelContext_1->globalParams_0->s0_0)[qo_0 >> 2U])) >> ((qo_0 & 3U) * 8U)) & 255U;

#line 44
    uint32_t nib_0;
    if(w_0 < 16U)
    {

#line 45
        nib_0 = qb_0 & 15U;

#line 45
    }
    else
    {

#line 45
        nib_0 = qb_0 >> 4U;

#line 45
    }
    return (float(nib_0) - 8.0f) * f16_bits_to_f32_0(dh_0);
}

static float ldb_0(uint32_t e_1, KernelContext_0 * kernelContext_2)
{

#line 50
    return *(&(kernelContext_2->globalParams_0->s1_0)[e_1]);
}


void _main_0(void* _S3, void* entryPointParams_0, void* globalParams_1)
{

#line 54
    ComputeThreadVaryingInput * _S4 = (slang_bit_cast<ComputeThreadVaryingInput *>(_S3));

#line 54
    KernelContext_0 kernelContext_3;

#line 54
    (&kernelContext_3)->globalParams_0 = (slang_bit_cast<GlobalParams_0*>(globalParams_1));

#line 54
    Vector<uint32_t, 3>  _S5 = _S4->groupID;

#line 54
    Vector<uint32_t, 3>  _S6 = _S4->groupThreadID;

#line 54
    uint32_t _S7 = pw_0(10U, &kernelContext_3);

#line 54
    uint32_t _S8 = pw_0(11U, &kernelContext_3);

#line 54
    uint32_t _S9 = pw_0(20U, &kernelContext_3);

#line 54
    uint32_t _S10 = pw_0(21U, &kernelContext_3);

#line 59
    uint32_t _S11 = _S5.z;

#line 59
    uint32_t b2_0 = _S11 % _S10;
    uint32_t b3_0 = _S11 / _S10;

#line 60
    uint32_t _S12 = pw_0(18U, &kernelContext_3);

#line 60
    uint32_t _S13 = pw_0(55U, &kernelContext_3);
    uint32_t _S14 = b2_0 / _S13;

#line 61
    uint32_t _S15 = pw_0(16U, &kernelContext_3);

#line 61
    uint32_t _S16 = _S12 + _S14 * _S15;

#line 61
    uint32_t _S17 = pw_0(56U, &kernelContext_3);

#line 61
    uint32_t _S18 = b3_0 / _S17;

#line 61
    uint32_t _S19 = pw_0(17U, &kernelContext_3);

#line 61
    uint32_t _S20 = _S16 + _S18 * _S19;

#line 61
    uint32_t _S21 = pw_0(27U, &kernelContext_3);

#line 61
    uint32_t _S22 = pw_0(25U, &kernelContext_3);
    uint32_t _S23 = _S21 + b2_0 * _S22;

#line 62
    uint32_t _S24 = pw_0(26U, &kernelContext_3);

#line 62
    uint32_t _S25 = _S23 + b3_0 * _S24;

#line 62
    uint32_t _S26 = pw_0(9U, &kernelContext_3);

#line 62
    uint32_t _S27 = pw_0(7U, &kernelContext_3);
    uint32_t _S28 = _S26 + b2_0 * _S27;

#line 63
    uint32_t _S29 = pw_0(8U, &kernelContext_3);

#line 63
    uint32_t dBase_0 = _S28 + b3_0 * _S29;

#line 63
    uint32_t _S30 = pw_0(14U, &kernelContext_3);

#line 63
    uint32_t _S31 = pw_0(15U, &kernelContext_3);

#line 63
    uint32_t _S32 = pw_0(23U, &kernelContext_3);

#line 63
    uint32_t _S33 = pw_0(24U, &kernelContext_3);

#line 63
    uint32_t _S34 = pw_0(5U, &kernelContext_3);

#line 63
    uint32_t _S35 = pw_0(6U, &kernelContext_3);

#line 70
    uint32_t _S36 = _S5.y;

#line 70
    uint32_t _S37 = pw_0(57U, &kernelContext_3);

#line 70
    uint32_t row_0 = (_S36 * _S37 + _S5.x) * 8U + _S6.y;

#line 70
    bool _S38;

#line 76
    if((_S6.x) != 0U)
    {

#line 76
        _S38 = true;

#line 76
    }
    else
    {

#line 76
        _S38 = row_0 >= _S8;

#line 76
    }

#line 76
    if(_S38)
    {

#line 77
        return;
    }
    FixedArray<float, 32>  q0_0;
    FixedArray<float, 32>  q1_0;
    FixedArray<float, 32>  q2_0;
    FixedArray<float, 32>  q3_0;

#line 82
    uint32_t l_0 = 0U;
    for(;;)
    {

#line 83
        if(l_0 < 32U)
        {
        }
        else
        {

#line 83
            break;
        }

#line 83
        uint32_t k_1 = l_0;

#line 83
        float p0_0 = 0.0f;

#line 83
        float p1_0 = 0.0f;

#line 83
        float p2_0 = 0.0f;

#line 83
        float p3_0 = 0.0f;

#line 89
        for(;;)
        {

#line 89
            if(k_1 < _S7)
            {
            }
            else
            {

#line 89
                break;
            }

#line 89
            float _S39 = lda_0(_S20 + k_1 * _S30 + row_0 * _S31, &kernelContext_3);

            uint32_t bo_1 = _S25 + k_1 * _S32;

#line 91
            float _S40 = ldb_0(bo_1, &kernelContext_3);
            float _S41 = p0_0 + _S39 * _S40;
            if(_S9 > 1U)
            {

#line 93
                float _S42 = ldb_0(bo_1 + _S33, &kernelContext_3);

#line 93
                p1_0 = p1_0 + _S39 * _S42;

#line 93
            }


            if(_S9 > 2U)
            {

#line 96
                float _S43 = ldb_0(bo_1 + 2U * _S33, &kernelContext_3);

#line 96
                p2_0 = p2_0 + _S39 * _S43;

#line 96
            }


            if(_S9 > 3U)
            {

#line 99
                float _S44 = ldb_0(bo_1 + 3U * _S33, &kernelContext_3);

#line 99
                p3_0 = p3_0 + _S39 * _S44;

#line 99
            }

#line 99
            k_1 = k_1 + 32U;

#line 99
            p0_0 = _S41;

#line 89
        }

#line 104
        q0_0[l_0] = p0_0;
        q1_0[l_0] = p1_0;
        q2_0[l_0] = p2_0;
        q3_0[l_0] = p3_0;

#line 83
        l_0 = l_0 + 1U;

#line 83
    }

#line 83
    l_0 = 0U;

#line 109
    for(;;)
    {

#line 109
        if(l_0 < 16U)
        {
        }
        else
        {

#line 109
            break;
        }

#line 110
        uint32_t _S45 = l_0 + 16U;

#line 110
        q0_0[l_0] = q0_0[l_0] + q0_0[_S45];
        q1_0[l_0] = q1_0[l_0] + q1_0[_S45];
        q2_0[l_0] = q2_0[l_0] + q2_0[_S45];
        q3_0[l_0] = q3_0[l_0] + q3_0[_S45];

#line 109
        l_0 = l_0 + 1U;

#line 109
    }

#line 109
    l_0 = 0U;

#line 115
    for(;;)
    {

#line 115
        if(l_0 < 8U)
        {
        }
        else
        {

#line 115
            break;
        }

#line 116
        uint32_t _S46 = l_0 + 8U;

#line 116
        q0_0[l_0] = q0_0[l_0] + q0_0[_S46];
        q1_0[l_0] = q1_0[l_0] + q1_0[_S46];
        q2_0[l_0] = q2_0[l_0] + q2_0[_S46];
        q3_0[l_0] = q3_0[l_0] + q3_0[_S46];

#line 115
        l_0 = l_0 + 1U;

#line 115
    }

#line 115
    l_0 = 0U;

#line 121
    for(;;)
    {

#line 121
        if(l_0 < 4U)
        {
        }
        else
        {

#line 121
            break;
        }

#line 122
        uint32_t _S47 = l_0 + 4U;

#line 122
        q0_0[l_0] = q0_0[l_0] + q0_0[_S47];
        q1_0[l_0] = q1_0[l_0] + q1_0[_S47];
        q2_0[l_0] = q2_0[l_0] + q2_0[_S47];
        q3_0[l_0] = q3_0[l_0] + q3_0[_S47];

#line 121
        l_0 = l_0 + 1U;

#line 121
    }

#line 121
    l_0 = 0U;

#line 127
    for(;;)
    {

#line 127
        if(l_0 < 2U)
        {
        }
        else
        {

#line 127
            break;
        }

#line 128
        uint32_t _S48 = l_0 + 2U;

#line 128
        q0_0[l_0] = q0_0[l_0] + q0_0[_S48];
        q1_0[l_0] = q1_0[l_0] + q1_0[_S48];
        q2_0[l_0] = q2_0[l_0] + q2_0[_S48];
        q3_0[l_0] = q3_0[l_0] + q3_0[_S48];

#line 127
        l_0 = l_0 + 1U;

#line 127
    }

#line 127
    l_0 = 0U;

#line 133
    for(;;)
    {

#line 133
        if(l_0 < 1U)
        {
        }
        else
        {

#line 133
            break;
        }

#line 134
        uint32_t _S49 = l_0 + 1U;

#line 134
        q0_0[l_0] = q0_0[l_0] + q0_0[_S49];
        q1_0[l_0] = q1_0[l_0] + q1_0[_S49];
        q2_0[l_0] = q2_0[l_0] + q2_0[_S49];
        q3_0[l_0] = q3_0[l_0] + q3_0[_S49];

#line 133
        l_0 = _S49;

#line 133
    }

#line 139
    if(_S9 > 0U)
    {

#line 140
        *(&((&kernelContext_3)->globalParams_0->dst_0)[dBase_0 + row_0 * _S34]) = q0_0[0U];

#line 139
    }


    if(_S9 > 1U)
    {

#line 143
        *(&((&kernelContext_3)->globalParams_0->dst_0)[dBase_0 + row_0 * _S34 + _S35]) = q1_0[0U];

#line 142
    }


    if(_S9 > 2U)
    {

#line 146
        *(&((&kernelContext_3)->globalParams_0->dst_0)[dBase_0 + row_0 * _S34 + 2U * _S35]) = q2_0[0U];

#line 145
    }


    if(_S9 > 3U)
    {

#line 149
        *(&((&kernelContext_3)->globalParams_0->dst_0)[dBase_0 + row_0 * _S34 + 3U * _S35]) = q3_0[0U];

#line 148
    }


    return;
}

// [numthreads(32, 8, 1)]
SLANG_PRELUDE_EXPORT
void main_0_Thread(ComputeThreadVaryingInput* varyingInput, void* entryPointParams, void* globalParams)
{
    _main_0(varyingInput, entryPointParams, globalParams);
}
// [numthreads(32, 8, 1)]
SLANG_PRELUDE_EXPORT
void main_0_Group(ComputeVaryingInput* varyingInput, void* entryPointParams, void* globalParams)
{
    ComputeThreadVaryingInput threadInput = {};
    threadInput.groupID = varyingInput->startGroupID;
    for (uint32_t y = 0; y < 8; ++y)
    {
        threadInput.groupThreadID.y = y;
        for (uint32_t x = 0; x < 32; ++x)
        {
            threadInput.groupThreadID.x = x;
            _main_0(&threadInput, entryPointParams, globalParams);
        }
    }
}
// [numthreads(32, 8, 1)]
SLANG_PRELUDE_EXPORT
void main_0(ComputeVaryingInput* varyingInput, void* entryPointParams, void* globalParams)
{
    ComputeVaryingInput vi = *varyingInput;
    ComputeVaryingInput groupVaryingInput = {};
    for (uint32_t z = vi.startGroupID.z; z < vi.endGroupID.z; ++z)
    {
        groupVaryingInput.startGroupID.z = z;
        for (uint32_t y = vi.startGroupID.y; y < vi.endGroupID.y; ++y)
        {
            groupVaryingInput.startGroupID.y = y;
            for (uint32_t x = vi.startGroupID.x; x < vi.endGroupID.x; ++x)
            {
                groupVaryingInput.startGroupID.x = x;
                main_0_Group(&groupVaryingInput, entryPointParams, globalParams);
            }
        }
    }
}
