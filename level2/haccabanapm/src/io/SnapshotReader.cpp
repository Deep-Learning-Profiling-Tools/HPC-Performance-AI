// src/io/SnapshotReader.cpp
//
// Parallel-HDF5 reader.  Reads metadata attributes from the root group,
// determines the global particle count from the position dataset's
// dataspace, partitions ranges across ranks, and reads each particle
// field as a hyperslab into a host buffer; the host buffer is then
// scattered back into the device AoSoA via a parallel_for following
// the prompt-08 SYCL device-first pattern.
//
// Schema versions:
//   v2 (current): includes /particles/phi and /particles/status, and the
//       ATTR_TOPOLOGY_DIMS metadata attribute.
//   v1 (prompt-11 era): no phi/status/topology_dims.  Reader transparently
//       backfills phi=0 and status=local_rank, leaves topology_dims=(0,0,0),
//       and logs a warning to stderr from rank 0.

#include "SnapshotReader.hpp"
#include "SnapshotSchema.hpp"
#include "../Particles.hpp"
#include "../Types.hpp"

#include <Kokkos_Core.hpp>
#include <hdf5.h>
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
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

void read_scalar_attr(hid_t loc, const char* name, hid_t dtype, void* dst)
{
    hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    check_h5_id(attr, (std::string("H5Aopen ") + name).c_str());
    check_h5(H5Aread(attr, dtype, dst), (std::string("H5Aread ") + name).c_str());
    check_h5(H5Aclose(attr), "H5Aclose");
}

bool attr_exists(hid_t loc, const char* name)
{
    htri_t r = H5Aexists(loc, name);
    return r > 0;
}

void read_attrs(hid_t file, SimulationState& s, int& schema_version_out)
{
    int sv = 0;
    read_scalar_attr(file, ATTR_SCHEMA_VERSION, H5T_NATIVE_INT, &sv);
    schema_version_out = sv;
    if (sv < 1 || sv > SCHEMA_VERSION) {
        throw std::runtime_error(
            "snapshot schema_version unsupported: file=" + std::to_string(sv) +
            " supported=[1," + std::to_string(SCHEMA_VERSION) + "]");
    }
    read_scalar_attr(file, ATTR_SCALE_FACTOR,   H5T_NATIVE_DOUBLE, &s.a);
    read_scalar_attr(file, ATTR_STEP,           H5T_NATIVE_INT,    &s.step);
    read_scalar_attr(file, ATTR_NG,             H5T_NATIVE_INT,    &s.ng);
    read_scalar_attr(file, ATTR_NP,             H5T_NATIVE_INT64,  &s.np);
    read_scalar_attr(file, ATTR_RL_MPC_H,       H5T_NATIVE_DOUBLE, &s.rL);
    read_scalar_attr(file, ATTR_Z_INIT,         H5T_NATIVE_DOUBLE, &s.z_in);
    read_scalar_attr(file, ATTR_OMEGA_M,        H5T_NATIVE_DOUBLE, &s.omega_m);
    read_scalar_attr(file, ATTR_OMEGA_CB,       H5T_NATIVE_DOUBLE, &s.omega_cb);
    read_scalar_attr(file, ATTR_OMEGA_B,        H5T_NATIVE_DOUBLE, &s.omega_b);
    read_scalar_attr(file, ATTR_H,              H5T_NATIVE_DOUBLE, &s.h);
    read_scalar_attr(file, ATTR_N_S,            H5T_NATIVE_DOUBLE, &s.n_s);
    read_scalar_attr(file, ATTR_SIGMA_8,        H5T_NATIVE_DOUBLE, &s.sigma_8);

    if (sv >= 2 && attr_exists(file, ATTR_TOPOLOGY_DIMS)) {
        // Read the 3-element int array.
        hid_t attr = H5Aopen(file, ATTR_TOPOLOGY_DIMS, H5P_DEFAULT);
        check_h5_id(attr, "H5Aopen topology_dims");
        check_h5(H5Aread(attr, H5T_NATIVE_INT, s.topology_dims),
                 "H5Aread topology_dims");
        check_h5(H5Aclose(attr), "H5Aclose topology_dims");
    } else {
        s.topology_dims[0] = s.topology_dims[1] = s.topology_dims[2] = 0;
    }
}

// Determine the leading-dimension extent of a dataset.  Used to discover
// the global particle count from the position dataset.
hsize_t get_dataset_leading_dim(hid_t file, const char* path)
{
    hid_t dset = H5Dopen2(file, path, H5P_DEFAULT);
    check_h5_id(dset, (std::string("H5Dopen2 ") + path).c_str());
    hid_t space = H5Dget_space(dset);
    check_h5_id(space, "H5Dget_space");
    int ndims = H5Sget_simple_extent_ndims(space);
    if (ndims < 1)
        throw std::runtime_error(std::string("dataset has no dims: ") + path);
    std::vector<hsize_t> dims(ndims);
    if (H5Sget_simple_extent_dims(space, dims.data(), nullptr) < 0)
        throw std::runtime_error("H5Sget_simple_extent_dims failed");
    H5Sclose(space);
    H5Dclose(dset);
    return dims[0];
}

void read_dataset(hid_t file, const char* name, hid_t dtype,
                  int ndims, const hsize_t* global_dims,
                  hsize_t local_count, hsize_t global_offset,
                  void* data)
{
    hid_t dset = H5Dopen2(file, name, H5P_DEFAULT);
    check_h5_id(dset, (std::string("H5Dopen2 ") + name).c_str());
    hid_t fspace = H5Dget_space(dset);
    check_h5_id(fspace, "H5Dget_space");

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

    check_h5(H5Dread(dset, dtype, mspace, fspace, dxpl, data), "H5Dread");

    check_h5(H5Pclose(dxpl), "H5Pclose(dxpl)");
    check_h5(H5Sclose(mspace), "H5Sclose(mspace)");
    check_h5(H5Sclose(fspace), "H5Sclose(fspace)");
    check_h5(H5Dclose(dset), "H5Dclose");
}

} // namespace

void read_snapshot(const std::string& filename,
                   Particles&         parts,
                   SimulationState&   state)
{
    MPI_Comm comm = parts.comm();
    int my_rank = 0, n_ranks = 1;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &n_ranks);

    // Open with parallel access.
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    check_h5_id(fapl, "H5Pcreate(FILE_ACCESS)");
    check_h5(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL),
             "H5Pset_fapl_mpio");
    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, fapl);
    check_h5_id(file, "H5Fopen");
    check_h5(H5Pclose(fapl), "H5Pclose(fapl)");

    // Metadata, schema version check.
    int schema_version_in_file = 0;
    read_attrs(file, state, schema_version_in_file);

    // Discover global N from the position dataset.
    const hsize_t n_global = get_dataset_leading_dim(file, PATH_POS);

    // Range partition: first (n_global % n_ranks) ranks get one extra.
    const hsize_t base   = n_global / static_cast<hsize_t>(n_ranks);
    const hsize_t extras = n_global % static_cast<hsize_t>(n_ranks);
    const hsize_t my_count = base + (static_cast<hsize_t>(my_rank) < extras ? 1 : 0);
    const hsize_t my_offset =
        base * static_cast<hsize_t>(my_rank) +
        std::min<hsize_t>(static_cast<hsize_t>(my_rank), extras);

    parts.resize(static_cast<std::size_t>(my_count));

    // Allocate host buffers (1-element placeholder for empty ranks).
    // LayoutRight matches HDF5's row-major hyperslab layout — see the
    // long comment in SnapshotWriter.cpp for why this is mandatory.
    const std::size_t n_alloc = (my_count > 0) ? my_count : 1;
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
    auto pos_h    = Kokkos::create_mirror_view(pos_d);
    auto vel_h    = Kokkos::create_mirror_view(vel_d);
    auto id_h     = Kokkos::create_mirror_view(id_d);
    auto mask_h   = Kokkos::create_mirror_view(mask_d);
    auto mass_h   = Kokkos::create_mirror_view(mass_d);
    auto phi_h    = Kokkos::create_mirror_view(phi_d);
    auto status_h = Kokkos::create_mirror_view(status_d);

    {
        hsize_t dims2[2] = { n_global, 3 };
        read_dataset(file, PATH_POS, H5T_NATIVE_FLOAT,
                     2, dims2, my_count, my_offset, pos_h.data());
        read_dataset(file, PATH_VEL, H5T_NATIVE_FLOAT,
                     2, dims2, my_count, my_offset, vel_h.data());
    }
    {
        hsize_t dims1[1] = { n_global };
        read_dataset(file, PATH_ID, H5T_NATIVE_INT64,
                     1, dims1, my_count, my_offset, id_h.data());
        read_dataset(file, PATH_MASK, H5T_NATIVE_UINT16,
                     1, dims1, my_count, my_offset, mask_h.data());
        read_dataset(file, PATH_MASS, H5T_NATIVE_FLOAT,
                     1, dims1, my_count, my_offset, mass_h.data());

        if (schema_version_in_file >= 2) {
            read_dataset(file, PATH_PHI, H5T_NATIVE_FLOAT,
                         1, dims1, my_count, my_offset, phi_h.data());
            read_dataset(file, PATH_STATUS, H5T_NATIVE_INT32,
                         1, dims1, my_count, my_offset, status_h.data());
        } else {
            // v1: backfill phi=0 and status=local_rank.  The status backfill
            // is the IC convention HACC uses (mc3.cxx:884 sets status to
            // Partition::getMyProc as a placeholder before updateStatus is
            // called); since v1 files predate the wrap_periodic migration,
            // the local rank is the best available proxy for "owning rank".
            for (hsize_t i = 0; i < my_count; ++i) {
                phi_h(i)    = 0.0f;
                status_h(i) = static_cast<int32_t>(my_rank);
            }
            if (my_rank == 0) {
                std::fprintf(stderr,
                    "[pmk::read_snapshot] WARNING: schema v1 file '%s' — "
                    "backfilling phi=0 and status=local_rank.\n",
                    filename.c_str());
            }
        }
    }
    check_h5(H5Fclose(file), "H5Fclose");

    if (my_count == 0)
        return;

    // Push host → device, then scatter into the AoSoA slices.
    Kokkos::deep_copy(pos_d,    pos_h);
    Kokkos::deep_copy(vel_d,    vel_h);
    Kokkos::deep_copy(id_d,     id_h);
    Kokkos::deep_copy(mask_d,   mask_h);
    Kokkos::deep_copy(mass_d,   mass_h);
    Kokkos::deep_copy(phi_d,    phi_h);
    Kokkos::deep_copy(status_d, status_h);

    auto pos_s  = parts.pos();
    auto vel_s  = parts.vel();
    auto id_s   = parts.id();
    auto mask_s = parts.mask();
    auto mass_s = parts.mass();
    auto phi_s  = parts.phi();
    // status is reduced to the AoSoA's phi/mask layout — pmkokkos doesn't
    // carry a per-particle status slice (it's a pure I/O convention), so
    // status_d is read but discarded after the metadata round-trip.
    (void)status_d;

    Kokkos::parallel_for(
        "snap_unstage", Kokkos::RangePolicy<ExecSpace>(0, my_count),
        KOKKOS_LAMBDA(const std::size_t i) {
            pos_s(i, 0) = pos_d(i, 0);
            pos_s(i, 1) = pos_d(i, 1);
            pos_s(i, 2) = pos_d(i, 2);
            vel_s(i, 0) = vel_d(i, 0);
            vel_s(i, 1) = vel_d(i, 1);
            vel_s(i, 2) = vel_d(i, 2);
            id_s(i)     = id_d(i);
            mask_s(i)   = mask_d(i);
            mass_s(i)   = mass_d(i);
            phi_s(i)    = phi_d(i);
        });
    Kokkos::fence();
}

} // namespace pmk
