#include "/home/setup/tools/slang-2026.20/include/slang-cpp-prelude.h"

#ifdef SLANG_PRELUDE_NAMESPACE
using namespace SLANG_PRELUDE_NAMESPACE;
#endif


#line 1 "slang/mul_mat_serial_tiled_q4_0_f32.slang"
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
    uint32_t m0_0 = _S5.x * 64U;
    uint32_t n0_0 = _S5.y * 64U;

#line 88
    uint32_t _S36 = _S6.x;

#line 88
    uint32_t i0_0 = m0_0 + _S36;
    uint32_t i1_0 = m0_0 + (_S36 + 16U);
    uint32_t i2_0 = m0_0 + (_S36 + 32U);
    uint32_t i3_0 = m0_0 + (_S36 + 48U);
    uint32_t _S37 = _S6.y;

#line 92
    uint32_t j0_0 = n0_0 + _S37;
    uint32_t j1_0 = n0_0 + (_S37 + 16U);
    uint32_t j2_0 = n0_0 + (_S37 + 32U);
    uint32_t j3_0 = n0_0 + (_S37 + 48U);

#line 95
    float c00_0 = 0.0f;

#line 95
    float c01_0 = 0.0f;

#line 95
    float c02_0 = 0.0f;

#line 95
    float c03_0 = 0.0f;

#line 95
    float c10_0 = 0.0f;

#line 95
    float c11_0 = 0.0f;

#line 95
    float c12_0 = 0.0f;

#line 95
    float c13_0 = 0.0f;

#line 95
    float c20_0 = 0.0f;

#line 95
    float c21_0 = 0.0f;

#line 95
    float c22_0 = 0.0f;

#line 95
    float c23_0 = 0.0f;

#line 95
    float c30_0 = 0.0f;

#line 95
    float c31_0 = 0.0f;

#line 95
    float c32_0 = 0.0f;

#line 95
    float c33_0 = 0.0f;

#line 95
    uint32_t k_1 = 0U;
    for(;;)
    {

#line 96
        if(k_1 < _S7)
        {
        }
        else
        {

#line 96
            break;
        }

#line 96
        float x0_0;

        if(i0_0 < _S8)
        {

#line 98
            float _S38 = lda_0(_S20 + k_1 * _S30 + i0_0 * _S31, &kernelContext_3);

#line 98
            x0_0 = _S38;

#line 98
        }
        else
        {

#line 98
            x0_0 = 0.0f;

#line 98
        }

#line 98
        float x1_0;



        if(i1_0 < _S8)
        {

#line 102
            float _S39 = lda_0(_S20 + k_1 * _S30 + i1_0 * _S31, &kernelContext_3);

#line 102
            x1_0 = _S39;

#line 102
        }
        else
        {

#line 102
            x1_0 = 0.0f;

#line 102
        }

#line 102
        float x2_0;



        if(i2_0 < _S8)
        {

#line 106
            float _S40 = lda_0(_S20 + k_1 * _S30 + i2_0 * _S31, &kernelContext_3);

#line 106
            x2_0 = _S40;

#line 106
        }
        else
        {

#line 106
            x2_0 = 0.0f;

#line 106
        }

#line 106
        float x3_0;



        if(i3_0 < _S8)
        {

#line 110
            float _S41 = lda_0(_S20 + k_1 * _S30 + i3_0 * _S31, &kernelContext_3);

#line 110
            x3_0 = _S41;

#line 110
        }
        else
        {

#line 110
            x3_0 = 0.0f;

#line 110
        }

#line 110
        float y0_0;



        if(j0_0 < _S9)
        {

#line 114
            float _S42 = ldb_0(_S25 + k_1 * _S32 + j0_0 * _S33, &kernelContext_3);

#line 114
            y0_0 = _S42;

#line 114
        }
        else
        {

#line 114
            y0_0 = 0.0f;

#line 114
        }

#line 114
        float y1_0;



        if(j1_0 < _S9)
        {

#line 118
            float _S43 = ldb_0(_S25 + k_1 * _S32 + j1_0 * _S33, &kernelContext_3);

#line 118
            y1_0 = _S43;

#line 118
        }
        else
        {

#line 118
            y1_0 = 0.0f;

#line 118
        }

#line 118
        float y2_0;



        if(j2_0 < _S9)
        {

#line 122
            float _S44 = ldb_0(_S25 + k_1 * _S32 + j2_0 * _S33, &kernelContext_3);

#line 122
            y2_0 = _S44;

#line 122
        }
        else
        {

#line 122
            y2_0 = 0.0f;

#line 122
        }

#line 122
        float y3_0;



        if(j3_0 < _S9)
        {

#line 126
            float _S45 = ldb_0(_S25 + k_1 * _S32 + j3_0 * _S33, &kernelContext_3);

#line 126
            y3_0 = _S45;

#line 126
        }
        else
        {

#line 126
            y3_0 = 0.0f;

#line 126
        }


        float _S46 = c00_0 + x0_0 * y0_0;
        float _S47 = c01_0 + x0_0 * y1_0;
        float _S48 = c02_0 + x0_0 * y2_0;
        float _S49 = c03_0 + x0_0 * y3_0;
        float _S50 = c10_0 + x1_0 * y0_0;
        float _S51 = c11_0 + x1_0 * y1_0;
        float _S52 = c12_0 + x1_0 * y2_0;
        float _S53 = c13_0 + x1_0 * y3_0;
        float _S54 = c20_0 + x2_0 * y0_0;
        float _S55 = c21_0 + x2_0 * y1_0;
        float _S56 = c22_0 + x2_0 * y2_0;
        float _S57 = c23_0 + x2_0 * y3_0;
        float _S58 = c30_0 + x3_0 * y0_0;
        float _S59 = c31_0 + x3_0 * y1_0;
        float _S60 = c32_0 + x3_0 * y2_0;
        float _S61 = c33_0 + x3_0 * y3_0;

#line 96
        uint32_t k_2 = k_1 + 1U;

#line 96
        c00_0 = _S46;

#line 96
        c01_0 = _S47;

#line 96
        c02_0 = _S48;

#line 96
        c03_0 = _S49;

#line 96
        c10_0 = _S50;

#line 96
        c11_0 = _S51;

#line 96
        c12_0 = _S52;

#line 96
        c13_0 = _S53;

#line 96
        c20_0 = _S54;

#line 96
        c21_0 = _S55;

#line 96
        c22_0 = _S56;

#line 96
        c23_0 = _S57;

#line 96
        c30_0 = _S58;

#line 96
        c31_0 = _S59;

#line 96
        c32_0 = _S60;

#line 96
        c33_0 = _S61;

#line 96
        k_1 = k_2;

#line 96
    }

#line 146
    bool _S62 = i0_0 < _S8;

#line 146
    bool _S63;

#line 146
    if(_S62)
    {

#line 146
        _S63 = j0_0 < _S9;

#line 146
    }
    else
    {

#line 146
        _S63 = false;

#line 146
    }

#line 146
    if(_S63)
    {

#line 147
        *(&((&kernelContext_3)->globalParams_0->dst_0)[dBase_0 + i0_0 * _S34 + j0_0 * _S35]) = c00_0;

#line 146
    }


    if(_S62)
    {

#line 149
        _S63 = j1_0 < _S9;

#line 149
    }
    else
    {

#line 149
        _S63 = false;

#line 149
    }

#line 149
    if(_S63)
    {

#line 150
        *(&((&kernelContext_3)->globalParams_0->dst_0)[dBase_0 + i0_0 * _S34 + j1_0 * _S35]) = c01_0;

#line 149
    }


    if(_S62)
    {

#line 152
        _S63 = j2_0 < _S9;

#line 152
    }
    else
    {

#line 152
        _S63 = false;

#line 152
    }

#line 152
    if(_S63)
    {

#line 153
        *(&((&kernelContext_3)->globalParams_0->dst_0)[dBase_0 + i0_0 * _S34 + j2_0 * _S35]) = c02_0;

#line 152
    }


    if(_S62)
    {

#line 155
        _S63 = j3_0 < _S9;

#line 155
    }
    else
    {

#line 155
        _S63 = false;

#line 155
    }

#line 155
    if(_S63)
    {

#line 156
        *(&((&kernelContext_3)->globalParams_0->dst_0)[dBase_0 + i0_0 * _S34 + j3_0 * _S35]) = c03_0;

#line 155
    }


    bool _S64 = i1_0 < _S8;

#line 158
    if(_S64)
    {

#line 158
        _S63 = j0_0 < _S9;

#line 158
    }
    else
    {

#line 158
        _S63 = false;

#line 158
    }

#line 158
    if(_S63)
    {

#line 159
        *(&((&kernelContext_3)->globalParams_0->dst_0)[dBase_0 + i1_0 * _S34 + j0_0 * _S35]) = c10_0;

#line 158
    }


    if(_S64)
    {

#line 161
        _S63 = j1_0 < _S9;

#line 161
    }
    else
    {

#line 161
        _S63 = false;

#line 161
    }

#line 161
    if(_S63)
    {

#line 162
        *(&((&kernelContext_3)->globalParams_0->dst_0)[dBase_0 + i1_0 * _S34 + j1_0 * _S35]) = c11_0;

#line 161
    }


    if(_S64)
    {

#line 164
        _S63 = j2_0 < _S9;

#line 164
    }
    else
    {

#line 164
        _S63 = false;

#line 164
    }

#line 164
    if(_S63)
    {

#line 165
        *(&((&kernelContext_3)->globalParams_0->dst_0)[dBase_0 + i1_0 * _S34 + j2_0 * _S35]) = c12_0;

#line 164
    }


    if(_S64)
    {

#line 167
        _S63 = j3_0 < _S9;

#line 167
    }
    else
    {

#line 167
        _S63 = false;

#line 167
    }

#line 167
    if(_S63)
    {

#line 168
        *(&((&kernelContext_3)->globalParams_0->dst_0)[dBase_0 + i1_0 * _S34 + j3_0 * _S35]) = c13_0;

#line 167
    }


    bool _S65 = i2_0 < _S8;

#line 170
    if(_S65)
    {

#line 170
        _S63 = j0_0 < _S9;

#line 170
    }
    else
    {

#line 170
        _S63 = false;

#line 170
    }

#line 170
    if(_S63)
    {

#line 171
        *(&((&kernelContext_3)->globalParams_0->dst_0)[dBase_0 + i2_0 * _S34 + j0_0 * _S35]) = c20_0;

#line 170
    }


    if(_S65)
    {

#line 173
        _S63 = j1_0 < _S9;

#line 173
    }
    else
    {

#line 173
        _S63 = false;

#line 173
    }

#line 173
    if(_S63)
    {

#line 174
        *(&((&kernelContext_3)->globalParams_0->dst_0)[dBase_0 + i2_0 * _S34 + j1_0 * _S35]) = c21_0;

#line 173
    }


    if(_S65)
    {

#line 176
        _S63 = j2_0 < _S9;

#line 176
    }
    else
    {

#line 176
        _S63 = false;

#line 176
    }

#line 176
    if(_S63)
    {

#line 177
        *(&((&kernelContext_3)->globalParams_0->dst_0)[dBase_0 + i2_0 * _S34 + j2_0 * _S35]) = c22_0;

#line 176
    }


    if(_S65)
    {

#line 179
        _S63 = j3_0 < _S9;

#line 179
    }
    else
    {

#line 179
        _S63 = false;

#line 179
    }

#line 179
    if(_S63)
    {

#line 180
        *(&((&kernelContext_3)->globalParams_0->dst_0)[dBase_0 + i2_0 * _S34 + j3_0 * _S35]) = c23_0;

#line 179
    }


    bool _S66 = i3_0 < _S8;

#line 182
    if(_S66)
    {

#line 182
        _S63 = j0_0 < _S9;

#line 182
    }
    else
    {

#line 182
        _S63 = false;

#line 182
    }

#line 182
    if(_S63)
    {

#line 183
        *(&((&kernelContext_3)->globalParams_0->dst_0)[dBase_0 + i3_0 * _S34 + j0_0 * _S35]) = c30_0;

#line 182
    }


    if(_S66)
    {

#line 185
        _S63 = j1_0 < _S9;

#line 185
    }
    else
    {

#line 185
        _S63 = false;

#line 185
    }

#line 185
    if(_S63)
    {

#line 186
        *(&((&kernelContext_3)->globalParams_0->dst_0)[dBase_0 + i3_0 * _S34 + j1_0 * _S35]) = c31_0;

#line 185
    }


    if(_S66)
    {

#line 188
        _S63 = j2_0 < _S9;

#line 188
    }
    else
    {

#line 188
        _S63 = false;

#line 188
    }

#line 188
    if(_S63)
    {

#line 189
        *(&((&kernelContext_3)->globalParams_0->dst_0)[dBase_0 + i3_0 * _S34 + j2_0 * _S35]) = c32_0;

#line 188
    }


    if(_S66)
    {

#line 191
        _S63 = j3_0 < _S9;

#line 191
    }
    else
    {

#line 191
        _S63 = false;

#line 191
    }

#line 191
    if(_S63)
    {

#line 192
        *(&((&kernelContext_3)->globalParams_0->dst_0)[dBase_0 + i3_0 * _S34 + j3_0 * _S35]) = c33_0;

#line 191
    }


    return;
}

// [numthreads(16, 16, 1)]
SLANG_PRELUDE_EXPORT
void main_0_Thread(ComputeThreadVaryingInput* varyingInput, void* entryPointParams, void* globalParams)
{
    _main_0(varyingInput, entryPointParams, globalParams);
}
// [numthreads(16, 16, 1)]
SLANG_PRELUDE_EXPORT
void main_0_Group(ComputeVaryingInput* varyingInput, void* entryPointParams, void* globalParams)
{
    ComputeThreadVaryingInput threadInput = {};
    threadInput.groupID = varyingInput->startGroupID;
    for (uint32_t y = 0; y < 16; ++y)
    {
        threadInput.groupThreadID.y = y;
        for (uint32_t x = 0; x < 16; ++x)
        {
            threadInput.groupThreadID.x = x;
            _main_0(&threadInput, entryPointParams, globalParams);
        }
    }
}
// [numthreads(16, 16, 1)]
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
