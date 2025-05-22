// Copyright (c) OpenMMLab. All rights reserved.
#include <stdio.h>
#include <stdlib.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "pytorch_cuda_helper.hpp"
#include "voxelization_cuda_kernel.cuh"

int HardVoxelizeForwardCUDAKernelLauncher(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim = 3) {
  // current version tooks about 0.04s for one frame on cpu
  // check device

  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int num_points = points.size(0);
  const int num_features = points.size(1);

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  // map points to voxel coors
  at::Tensor temp_coors =
      at::zeros({num_points, NDim}, points.options().dtype(at::kInt));

  dim3 grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
  dim3 block(512);

  // 1. link point to corresponding voxel coors
  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "hard_voxelize_kernel", ([&] {
        dynamic_voxelize_kernel<scalar_t, int><<<grid, block, 0, stream>>>(
            points.contiguous().data_ptr<scalar_t>(),
            temp_coors.contiguous().data_ptr<int>(), voxel_x, voxel_y, voxel_z,
            coors_x_min, coors_y_min, coors_z_min, coors_x_max, coors_y_max,
            coors_z_max, grid_x, grid_y, grid_z, num_points, num_features,
            NDim);
      }));

  AT_CUDA_CHECK(cudaGetLastError());

  // 2. map point to the idx of the corresponding voxel, find duplicate coor
  // create some temporary variables
  auto point_to_pointidx = -at::ones(
      {
          num_points,
      },
      points.options().dtype(at::kInt));
  auto point_to_voxelidx = -at::ones(
      {
          num_points,
      },
      points.options().dtype(at::kInt));

  dim3 map_grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
  dim3 map_block(512);
 
  int64_t total_size = num_points * NDim * temp_coors.element_size();
  if (total_size > (1ULL << 30)) {
  AT_DISPATCH_ALL_TYPES(
      temp_coors.scalar_type(), "point_to_voxelidx", ([&] {
        point_to_voxelidx_kernel<int, 512><<<map_grid, map_block, 0, stream>>>(
            temp_coors.contiguous().data_ptr<int>(),
            point_to_voxelidx.contiguous().data_ptr<int>(),
            point_to_pointidx.contiguous().data_ptr<int>(), max_points,
            max_voxels, num_points, NDim);
      }));
  } else if (total_size <= (1ULL << 30) && NDim == 3) {
    // 1. prepare data
    thrust::device_vector<PointCoord<int>> coords(num_points);
    auto coords_ptr = thrust::device_pointer_cast(coords.data());
    auto coor_ptr = thrust::raw_pointer_cast(temp_coors.contiguous().data_ptr<int>());

    thrust::transform(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_points),
        coords_ptr,
        CoordTransformer<int>{coor_ptr}
    );
    // 2. sort
    thrust::sort(thrust::device, coords.begin(), coords.end());
    // 3. point to voxel
    AT_DISPATCH_ALL_TYPES(
        temp_coors.scalar_type(), "point_to_voxelidx_with_sort", ([&] {
            point_to_voxelidx_with_sort_kernel<int><<<map_grid, map_block, 0, stream>>>(
                thrust::raw_pointer_cast(coords.data()),
                point_to_voxelidx.contiguous().data_ptr<int>(),
                point_to_pointidx.contiguous().data_ptr<int>(),
                max_points, max_voxels, num_points);
        }));
  }

  AT_CUDA_CHECK(cudaGetLastError());

  // 3. determine voxel num and voxel's coor index
  // make the logic in the CUDA device could accelerate about 10 times
  auto coor_to_voxelidx = -at::ones(
      {
          num_points,
      },
      points.options().dtype(at::kInt));
//   auto voxel_num = at::zeros(
//       {
//           1,
//       },
//       points.options().dtype(at::kInt));  // must be zero from the beginning

  dim3 determin_grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
  dim3 determin_block(512);
  // i. get if point need update voxel
  thrust::device_vector<bool> need_new_voxel(num_points);
  AT_DISPATCH_ALL_TYPES(temp_coors.scalar_type(), "determin_mark_new_voxels", ([&] {
                            mark_new_voxels<<<determin_grid, determin_block, 0, stream>>>(
                                point_to_voxelidx.contiguous().data_ptr<int>(),
                                need_new_voxel.data().get(), num_points
                            );
                        }));
  // ii. exclusive scan
  thrust::device_vector<int> voxel_indices(num_points);
  thrust::exclusive_scan(need_new_voxel.begin(), need_new_voxel.end(), voxel_indices.begin(), 0);
  // iii. update voxels
  AT_DISPATCH_ALL_TYPES(temp_coors.scalar_type(), "determin_update_voxels", ([&] {
                            update_voxels<<<determin_grid, determin_block, 0, stream>>>(
                                num_points_per_voxel.contiguous().data_ptr<int>(),
                                coor_to_voxelidx.contiguous().data_ptr<int>(),
                                point_to_voxelidx.contiguous().data_ptr<int>(),
                                voxel_indices.data().get(),
                                max_voxels, num_points
                            );
                        }));
  // IV. determin_voxel_num
  AT_DISPATCH_ALL_TYPES(temp_coors.scalar_type(), "determin_duplicate", ([&] {
                          determin_voxel_num<int><<<determin_grid, determin_block, 0, stream>>>(
                              num_points_per_voxel.contiguous().data_ptr<int>(),
                              point_to_voxelidx.contiguous().data_ptr<int>(),
                              point_to_pointidx.contiguous().data_ptr<int>(),
                              coor_to_voxelidx.contiguous().data_ptr<int>(),
                              voxel_indices.data().get(),
                              max_points, max_voxels, num_points);
                        }));

  AT_CUDA_CHECK(cudaGetLastError());

  // 4. copy point features to voxels
  // Step 4 & 5 could be parallel
  auto pts_output_size = num_points * num_features;
  dim3 cp_grid(std::min(at::cuda::ATenCeilDiv(pts_output_size, 512), 4096));
  dim3 cp_block(512);
  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "assign_point_to_voxel", ([&] {
        assign_point_to_voxel<float, int><<<cp_grid, cp_block, 0, stream>>>(
            pts_output_size, points.contiguous().data_ptr<float>(),
            point_to_voxelidx.contiguous().data_ptr<int>(),
            coor_to_voxelidx.contiguous().data_ptr<int>(),
            voxels.contiguous().data_ptr<float>(), max_points, num_features,
            num_points, NDim);
      }));
  //   cudaDeviceSynchronize();
  //   AT_CUDA_CHECK(cudaGetLastError());

  // 5. copy coors of each voxels
  auto coors_output_size = num_points * NDim;
  dim3 coors_cp_grid(
      std::min(at::cuda::ATenCeilDiv(coors_output_size, 512), 4096));
  dim3 coors_cp_block(512);
  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "assign_point_to_voxel", ([&] {
        assign_voxel_coors<float, int>
            <<<coors_cp_grid, coors_cp_block, 0, stream>>>(
                coors_output_size, temp_coors.contiguous().data_ptr<int>(),
                point_to_voxelidx.contiguous().data_ptr<int>(),
                coor_to_voxelidx.contiguous().data_ptr<int>(),
                coors.contiguous().data_ptr<int>(), num_points, NDim);
      }));

  AT_CUDA_CHECK(cudaGetLastError());

  int voxel_indices_last = 0;
  int need_new_voxel_last = 0;
  thrust::copy(voxel_indices.end() - 1, voxel_indices.end(), &voxel_indices_last);
  thrust::copy(need_new_voxel.end() - 1, need_new_voxel.end(), &need_new_voxel_last);
  int voxel_num_int = voxel_indices_last + need_new_voxel_last;

  return voxel_num_int;
}

int NondeterministicHardVoxelizeForwardCUDAKernelLauncher(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim = 3) {
  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int num_points = points.size(0);
  const int num_features = points.size(1);

  if (num_points == 0) return 0;

  dim3 blocks(
      std::min(at::cuda::ATenCeilDiv(num_points, THREADS_PER_BLOCK), 4096));
  dim3 threads(THREADS_PER_BLOCK);

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  // map points to voxel coors
  at::Tensor temp_coors =
      at::zeros({num_points, NDim}, points.options().dtype(at::kInt));

  // 1. link point to corresponding voxel coors
  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "hard_voxelize_kernel", ([&] {
        dynamic_voxelize_kernel<scalar_t, int><<<blocks, threads, 0, stream>>>(
            points.contiguous().data_ptr<scalar_t>(),
            temp_coors.contiguous().data_ptr<int>(), voxel_x, voxel_y, voxel_z,
            coors_x_min, coors_y_min, coors_z_min, coors_x_max, coors_y_max,
            coors_z_max, grid_x, grid_y, grid_z, num_points, num_features,
            NDim);
      }));

  at::Tensor coors_map;
  at::Tensor reduce_count;

  auto coors_clean = temp_coors.masked_fill(temp_coors.lt(0).any(-1, true), -1);

  std::tie(temp_coors, coors_map, reduce_count) =
      at::unique_dim(coors_clean, 0, true, true, false);

  if (temp_coors[0][0].lt(0).item<bool>()) {
    // the first element of temp_coors is (-1,-1,-1) and should be removed
    temp_coors = temp_coors.slice(0, 1);
    coors_map = coors_map - 1;
  }

  int num_coors = temp_coors.size(0);
  temp_coors = temp_coors.to(at::kInt);
  coors_map = coors_map.to(at::kInt);

  at::Tensor coors_count = at::zeros({1}, coors_map.options());
  at::Tensor coors_order = at::empty({num_coors}, coors_map.options());
  at::Tensor pts_id = at::zeros({num_points}, coors_map.options());
  reduce_count = at::zeros({num_coors}, coors_map.options());

  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "get_assign_pos", ([&] {
        nondeterministic_get_assign_pos<<<blocks, threads, 0, stream>>>(
            num_points, coors_map.contiguous().data_ptr<int32_t>(),
            pts_id.contiguous().data_ptr<int32_t>(),
            coors_count.contiguous().data_ptr<int32_t>(),
            reduce_count.contiguous().data_ptr<int32_t>(),
            coors_order.contiguous().data_ptr<int32_t>());
      }));

  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "assign_point_to_voxel", ([&] {
        nondeterministic_assign_point_voxel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                num_points, points.contiguous().data_ptr<scalar_t>(),
                coors_map.contiguous().data_ptr<int32_t>(),
                pts_id.contiguous().data_ptr<int32_t>(),
                temp_coors.contiguous().data_ptr<int32_t>(),
                reduce_count.contiguous().data_ptr<int32_t>(),
                coors_order.contiguous().data_ptr<int32_t>(),
                voxels.contiguous().data_ptr<scalar_t>(),
                coors.contiguous().data_ptr<int32_t>(),
                num_points_per_voxel.contiguous().data_ptr<int32_t>(),
                max_voxels, max_points, num_features, NDim);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
  return max_voxels < num_coors ? max_voxels : num_coors;
}

void DynamicVoxelizeForwardCUDAKernelLauncher(
    const at::Tensor &points, at::Tensor &coors,
    const std::vector<float> voxel_size, const std::vector<float> coors_range,
    const int NDim = 3) {
  // current version tooks about 0.04s for one frame on cpu
  // check device

  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int num_points = points.size(0);
  const int num_features = points.size(1);

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  const int col_blocks = at::cuda::ATenCeilDiv(num_points, THREADS_PER_BLOCK);
  dim3 blocks(col_blocks);
  dim3 threads(THREADS_PER_BLOCK);

  AT_DISPATCH_ALL_TYPES(points.scalar_type(), "dynamic_voxelize_kernel", [&] {
    dynamic_voxelize_kernel<scalar_t, int><<<blocks, threads, 0, stream>>>(
        points.contiguous().data_ptr<scalar_t>(),
        coors.contiguous().data_ptr<int>(), voxel_x, voxel_y, voxel_z,
        coors_x_min, coors_y_min, coors_z_min, coors_x_max, coors_y_max,
        coors_z_max, grid_x, grid_y, grid_z, num_points, num_features, NDim);
  });

  AT_CUDA_CHECK(cudaGetLastError());
}
