__kernel void conv(__global const float* x, const int x_len,
                  __constant const float* h, const int h_len,
                  __global float* y, __local float* localBuffer)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    int group_id = get_group_id(0);
	if (gid >= (x_len+h_len-1))
	{
		return;
	}
    int n = h_len - 1;

    int halo_left_index = group_id *local_size - n+lid;
	if (lid<n)
	{
		localBuffer[lid]= (halo_left_index < 0) ? 0 : x[halo_left_index];
	}
	if (gid < x_len)
	{
		localBuffer[n + lid] = x[gid];
	}
	else
	{
		localBuffer[n + lid] = 0;
	}
    barrier(CLK_LOCAL_MEM_FENCE);

    float temp = 0.0;
    for(int j = 0; j < h_len; j++)
    {
        temp += h[h_len - j - 1] * localBuffer[lid + j];
    }

    y[gid] = temp;

    return;
}

