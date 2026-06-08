#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// neg
//------------------------------------------------------------------------------
kernel void kernel_neg_f32(
    global void * src0_base,
    ulong offset0,
    global void * dst_base,
    ulong offsetd,
    int ne00, int ne01, int ne02, int ne03,
    ulong nb00, ulong nb01, ulong nb02, ulong nb03,
    int ne10, int ne11, int ne12, int ne13,
    ulong nb10, ulong nb11, ulong nb12, ulong nb13
) {
    const int i0 = get_global_id(0);
    const int i1 = get_global_id(1);
    const int i2 = get_global_id(2);

    if (i0 >= ne10 || i1 >= ne11 || i2 >= ne12) {
        return;
    }

    for (int i3 = 0; i3 < ne13; ++i3) {
        const ulong src_offset = (ulong)i0*nb00 + (ulong)i1*nb01 + (ulong)i2*nb02 + (ulong)i3*nb03;
        const ulong dst_offset = (ulong)i0*nb10 + (ulong)i1*nb11 + (ulong)i2*nb12 + (ulong)i3*nb13;

        global const float * src = (global const float *)((global char *)src0_base + offset0 + src_offset);
        global float *       dst = (global float *)((global char *)dst_base  + offsetd + dst_offset);

        *dst = -*src;
    }
}
