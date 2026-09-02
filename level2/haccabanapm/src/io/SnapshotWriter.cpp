// src/io/SnapshotWriter.cpp
//
// Parallel-HDF5 writer.  Layout:
//   - One global dataset per particle field (position/velocity/id/mask/mass/
//     phi/status).  phi and status are added in schema v2.
//   - Each rank stages its owned particles into a contiguous host buffer
//     and writes its hyperslab with H5FD_MPIO_COLLECTIVE.
//   - Metadata attributes live on the root group, including a 3-element
//     `topology_dims` array for v2 (the writer's MPI Cartesian dims).
//
// Device → host staging follows the prompt-08 SYCL pattern: allocate a
// contiguous device View, parallel_for to copy from the AoSoA slice,
// then create_mirror_view of the device View and deep_copy to host.
// Going device-first avoids the "no available copy mechanism" error
// triggered by the cabana_layout on PVC.

#include "SnapshotWriter.hpp"
#include "SnapshotSchema.hpp"
#include "../Particles.hpp"
#include "../Types.hpp"

#include <Kokkos_Core.hpp>
#include <hdf5.h>
#include <mpi.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace pmk {

namespace {

void check_h5(herr_t status, const char* what)
{
    if (status < 0)
        throw std::runtime_error(std::string("HDF5 error in ") + what);
}

void check_h5_id(hid_t id, const char* what)
{
    if (id < 0)
        throw std::runtime_error(std::string("HDF5 error in ") + what);
}

// Write one global dataset with collective MPI-IO.  Every rank participates;
// ranks with local_count==0 still call the collective write but with an
// H5S_NULL memory space and an H5S_NONE selection on the file space — HDF5
// requires this to keep the collective contract.
void write_dataset(hid_t file, const char* name, hid_t dtype,
                   int ndims, const hsize_t* global_dims,
                   hsize_t local_count, hsize_t global_offset,
                   const void* data)
{
    hid_t fspace = H5Screate_simple(ndims, global_dims, nullptr);
    check_h5_id(fspace, "H5Screate_simple (file)");
    hid_t dset = H5Dcreate2(file, name, dtype, fspace,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    check_h5_id(dset, "H5Dcreate2");

    if (local_count == 0) {
        check_h5(H5Sselect_none(fspace), "H5Sselect_none (file)");
    } else {
        std::vector<hsize_t> start(ndims, 0);
        std::vector<hsize_t> count(global_dims, global_dims + ndims);
        start[0] = global_offset;
        count[0] = local_count;
        check_h5(H5Sselect_hyperslab(fspace, H5S_SELECT_SET,
                                     start.data(), nullptr,
                                     count.data(), nullptr),
                 "H5Sselect_hyperslab (file)");
    }

    hid_t mspace;
    if (local_count == 0) {
        mspace = H5Screate(H5S_NULL);
    } else {
        std::vector<hsize_t> mdims(global_dims, global_dims + ndims);
        mdims[0] = local_count;
        mspace = H5Screate_simple(ndims, mdims.data(), nullptr);
    }
    check_h5_id(mspace, "H5Screate (mem)");

    hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
    check_h5_id(dxpl, "H5Pcreate(DATASET_XFER)");
    check_h5(H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE),
             "H5Pset_dxpl_mpio");

    check_h5(H5Dwrite(dset, dtype, mspace, fspace, dxpl, data),
             "H5Dwrite");

    check_h5(H5Pclose(dxpl), "H5Pclose(dxpl)");
    check_h5(H5Sclose(mspace), "H5Sclose(mspace)");
    check_h5(H5Dclose(dset), "H5Dclose");
    check_h5(H5Sclose(fspace), "H5Sclose(fspace)");
}

void write_scalar_attr(hid_t loc, const char* name, hid_t dtype,
                       const void* value)
{
    hid_t space = H5Screate(H5S_SCALAR);
    check_h5_id(space, "H5Screate(SCALAR)");
    hid_t attr = H5Acreate2(loc, name, dtype, space,
                            H5P_DEFAULT, H5P_DEFAULT);
    check_h5_id(attr, "H5Acreate2");
    check_h5(H5Awrite(attr, dtype, value), "H5Awrite");
    check_h5(H5Aclose(attr), "H5Aclose");
    check_h5(H5Sclose(space), "H5Sclose(scalar)");
}

void write_array_attr(hid_t loc, const char* name, hid_t dtype,
                      hsize_t n, const void* value)
{
    hid_t space = H5Screate_simple(1, &n, nullptr);
    check_h5_id(space, "H5Screate_simple(attr)");
    hid_t attr = H5Acreate2(loc, name, dtype, space,
                            H5P_DEFAULT, H5P_DEFAULT);
    check_h5_id(attr, "H5Acreate2(array)");
    check_h5(H5Awrite(attr, dtype, value), "H5Awrite(array)");
    check_h5(H5Aclose(attr), "H5Aclose(array)");
    check_h5(H5Sclose(space), "H5Sclose(array)");
}

void write_attrs(hid_t file, const SimulationState& s)
{
    int sv = SCHEMA_VERSION;
    write_scalar_attr(file, ATTR_SCHEMA_VERSION, H5T_NATIVE_INT,    &sv);
    write_scalar_attr(file, ATTR_SCALE_FACTOR,   H5T_NATIVE_DOUBLE, &s.a);
    write_scalar_attr(file, ATTR_STEP,           H5T_NATIVE_INT,    &s.step);
    write_scalar_attr(file, ATTR_NG,             H5T_NATIVE_INT,    &s.ng);
    write_scalar_attr(file, ATTR_NP,             H5T_NATIVE_INT64,  &s.np);
    write_scalar_attr(file, ATTR_RL_MPC_H,       H5T_NATIVE_DOUBLE, &s.rL);
    write_scalar_attr(file, ATTR_Z_INIT,         H5T_NATIVE_DOUBLE, &s.z_in);
    write_scalar_attr(file, ATTR_OMEGA_M,        H5T_NATIVE_DOUBLE, &s.omega_m);
    write_scalar_attr(file, ATTR_OMEGA_CB,       H5T_NATIVE_DOUBLE, &s.omega_cb);
    write_scalar_attr(file, ATTR_OMEGA_B,        H5T_NATIVE_DOUBLE, &s.omega_b);
    write_scalar_attr(file, ATTR_H,              H5T_NATIVE_DOUBLE, &s.h);
    write_scalar_attr(file, ATTR_N_S,            H5T_NATIVE_DOUBLE, &s.n_s);
    write_scalar_attr(file, ATTR_SIGMA_8,        H5T_NATIVE_DOUBLE, &s.sigma_8);
    write_array_attr (file, ATTR_TOPOLOGY_DIMS,  H5T_NATIVE_INT,
                      3, s.topology_dims);
}

} // namespace

void write_snapshot(const std::string&     filename,
                    Particles&             parts,
                    const SimulationState& state)
{
    MPI_Comm comm = parts.comm();
    int my_rank = 0, n_ranks = 1;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &n_ranks);

    const std::size_t n_local = parts.num_local();

    // Global N and exclusive-scan offset.
    long long n_local_ll  = static_cast<long long>(n_local);
    long long n_global_ll = 0;
    MPI_Allreduce(&n_local_ll, &n_global_ll, 1, MPI_LONG_LONG,
                  MPI_SUM, comm);
    long long offset_ll = 0;
    MPI_Exscan(&n_local_ll, &offset_ll, 1, MPI_LONG_LONG,
               MPI_SUM, comm);
    if (my_rank == 0) offset_ll = 0; // MPI_Exscan leaves rank 0 undefined.

    const hsize_t n_global = static_cast<hsize_t>(n_global_ll);
    const hsize_t offset   = static_cast<hsize_t>(offset_ll);

    // Stage device → host as contiguous LayoutRight buffers.  Allocate
    // device-first per the prompt-08 SYCL pattern, then deep_copy through
    // a HostMirror.  Sizes of zero get a 1-element placeholder so the
    // mirror_view is well-defined; the H5Dwrite for those ranks uses the
    // null-selection path inside write_dataset (data pointer ignored).
    //
    // LayoutRight is mandatory here: the SYCL default View layout is
    // LayoutLeft (column-major), and HDF5 expects row-major contiguous
    // storage for a (n, 3) dataset.  A LayoutLeft HostMirror would put
    // pos[axis*n + i] on disk and the multi-rank read with a different
    // partition would scramble axes.  Forcing LayoutRight makes
    // pos.data()[i*3 + axis] match HDF5's hyperslab semantics.
    const std::size_t n_alloc = (n_local > 0) ? n_local : 1;

    using Vec3D = Kokkos::View<float * [3], Kokkos::LayoutRight, MemorySpace>;
    using Vec1Df = Kokkos::View<float*,     Kokkos::LayoutRight, MemorySpace>;
    using I64D   = Kokkos::View<int64_t*,   Kokkos::LayoutRight, MemorySpace>;
    using I32D   = Kokkos::View<int32_t*,   Kokkos::LayoutRight, MemorySpace>;
    using U16D   = Kokkos::View<uint16_t*,  Kokkos::LayoutRight, MemorySpace>;
    Vec3D pos_d   ("snap_pos_d",    n_alloc);
    Vec3D vel_d   ("snap_vel_d",    n_alloc);
    I64D  id_d    ("snap_id_d",     n_alloc);
    U16D  mask_d  ("snap_mask_d",   n_alloc);
    Vec1Df mass_d ("snap_mass_d",   n_alloc);
    Vec1Df phi_d  ("snap_phi_d",    n_alloc);
    I32D  status_d("snap_status_d", n_alloc);

    // status is purely metadata — the local rank id (HACC sets it to the
    // owning rank in the 3D Cartesian topology after migration; our writer
    // requires the caller to have already migrated, so local rank is exactly
    // the owning rank).  Per hacc_ic_unit_conversions.md §5.
    const int32_t status_value = static_cast<int32_t>(my_rank);

    if (n_local > 0) {
        auto pos_s  = parts.pos();
        auto vel_s  = parts.vel();
        auto id_s   = parts.id();
        auto mask_s = parts.mask();
        auto mass_s = parts.mass();
        auto phi_s  = parts.phi();
        Kokkos::parallel_for(
            "snap_stage", Kokkos::RangePolicy<ExecSpace>(0, n_local),
            KOKKOS_LAMBDA(const std::size_t i) {
                pos_d(i, 0) = pos_s(i, 0);
                pos_d(i, 1) = pos_s(i, 1);
                pos_d(i, 2) = pos_s(i, 2);
                vel_d(i, 0) = vel_s(i, 0);
                vel_d(i, 1) = vel_s(i, 1);
                vel_d(i, 2) = vel_s(i, 2);
                id_d(i)     = id_s(i);
                mask_d(i)   = mask_s(i);
                mass_d(i)   = mass_s(i);
                phi_d(i)    = phi_s(i);
                status_d(i) = status_value;
            });
        Kokkos::fence();
    }

    auto pos_h    = Kokkos::create_mirror_view(pos_d);
    auto vel_h    = Kokkos::create_mirror_view(vel_d);
    auto id_h     = Kokkos::create_mirror_view(id_d);
    auto mask_h   = Kokkos::create_mirror_view(mask_d);
    auto mass_h   = Kokkos::create_mirror_view(mass_d);
    auto phi_h    = Kokkos::create_mirror_view(phi_d);
    auto status_h = Kokkos::create_mirror_view(status_d);
    Kokkos::deep_copy(pos_h,    pos_d);
    Kokkos::deep_copy(vel_h,    vel_d);
    Kokkos::deep_copy(id_h,     id_d);
    Kokkos::deep_copy(mask_h,   mask_d);
    Kokkos::deep_copy(mass_h,   mass_d);
    Kokkos::deep_copy(phi_h,    phi_d);
    Kokkos::deep_copy(status_h, status_d);

    // File creation with parallel access.
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    check_h5_id(fapl, "H5Pcreate(FILE_ACCESS)");
    check_h5(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL),
             "H5Pset_fapl_mpio");
    hid_t file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC,
                           H5P_DEFAULT, fapl);
    check_h5_id(file, "H5Fcreate");
    check_h5(H5Pclose(fapl), "H5Pclose(fapl)");

    // /particles group.
    hid_t grp = H5Gcreate2(file, "/particles", H5P_DEFAULT,
                           H5P_DEFAULT, H5P_DEFAULT);
    check_h5_id(grp, "H5Gcreate2(/particles)");
    check_h5(H5Gclose(grp), "H5Gclose(/particles)");

    {
        hsize_t dims2[2] = { n_global, 3 };
        write_dataset(file, PATH_POS, H5T_NATIVE_FLOAT,
                      2, dims2, n_local, offset, pos_h.data());
        write_dataset(file, PATH_VEL, H5T_NATIVE_FLOAT,
                      2, dims2, n_local, offset, vel_h.data());
    }
    {
        hsize_t dims1[1] = { n_global };
        write_dataset(file, PATH_ID, H5T_NATIVE_INT64,
                      1, dims1, n_local, offset, id_h.data());
        write_dataset(file, PATH_MASK, H5T_NATIVE_UINT16,
                      1, dims1, n_local, offset, mask_h.data());
        write_dataset(file, PATH_MASS, H5T_NATIVE_FLOAT,
                      1, dims1, n_local, offset, mass_h.data());
        write_dataset(file, PATH_PHI, H5T_NATIVE_FLOAT,
                      1, dims1, n_local, offset, phi_h.data());
        write_dataset(file, PATH_STATUS, H5T_NATIVE_INT32,
                      1, dims1, n_local, offset, status_h.data());
    }

    // Metadata.  Always overwrite np with the actually-written global
    // count to prevent a metadata/data mismatch.  topology_dims comes from
    // the caller (filled from MPI_Cart_get of parts.comm()) and is written
    // verbatim — see hacc_ic_unit_conversions.md §5 for why this matters.
    SimulationState s_out = state;
    s_out.np = static_cast<int64_t>(n_global_ll);
    write_attrs(file, s_out);

    check_h5(H5Fclose(file), "H5Fclose");
}

} // namespace pmk
