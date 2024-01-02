__kernel void conv(__global const float2* x, const int x_len,
                  __global const float2* h, const int h_len,
                  __global float2* y, __local float2* localBuffer)
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

    int halo_left_index = group_id * local_size - n + lid;
    if(lid < n)
    {
        localBuffer[lid].x = (halo_left_index < 0) ? 0 : x[halo_left_index].x;
        localBuffer[lid].y = (halo_left_index < 0) ? 0 : x[halo_left_index].y;
    }

	if (gid < x_len)
    {
        localBuffer[n + lid].x = x[gid].x;
        localBuffer[n + lid].y = x[gid].y;
    }
    else
	{
		localBuffer[n + lid].x = 0;
		localBuffer[n + lid].y = 0;
	}

    barrier(CLK_LOCAL_MEM_FENCE);

    float2 temp;
    temp.x = 0.0; temp.y = 0.0;

    for(int j = 0; j < h_len; j++)
    {
        temp.x += h[h_len - j - 1].x * localBuffer[lid + j].x - h[h_len - j - 1].y * localBuffer[lid + j].y;
        temp.y += h[h_len - j - 1].x * localBuffer[lid + j].y + h[h_len - j - 1].y * localBuffer[lid + j].x;
    }

    y[gid].x = temp.x;
    y[gid].y = temp.y;

    return;
}

