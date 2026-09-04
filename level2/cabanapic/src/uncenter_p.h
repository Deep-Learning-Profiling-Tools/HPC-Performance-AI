#ifndef uncenter_h
#define uncenter_h

void uncenter_particles(
        particle_list_t particles,
        interpolator_array_t& f0,
        real_t qdt_2mc
    )
{

    auto position_x = Cabana::slice<PositionX>(particles);
    auto position_y = Cabana::slice<PositionY>(particles);
    auto position_z = Cabana::slice<PositionZ>(particles);

    auto velocity_x = Cabana::slice<VelocityX>(particles);
    auto velocity_y = Cabana::slice<VelocityY>(particles);
    auto velocity_z = Cabana::slice<VelocityZ>(particles);

    //auto weight = Cabana::slice<Weight>(particles);
    auto cell = Cabana::slice<Cell_Index>(particles);

    const real_t qdt_4mc        = -0.5*qdt_2mc; // For backward half rotate
    const real_t one            = 1.;
    const real_t one_third      = 1./3.;
    const real_t two_fifteenths = 2./15.;

    // HPC-Performance-AI: Cabana::slice() is a host-only function; calling it
    // inside the KOKKOS_LAMBDA below (as upstream did, see its "hoist slice
    // call?" TODO) is diagnosed by nvcc 13.2 as 18 "calling a __host__ function
    // from a __host__ __device__ function" warnings and cannot run on the
    // device. The slices are taken once here instead, exactly as push.h does.
    auto _ex       = Cabana::slice<EX>(f0);
    auto _dexdy    = Cabana::slice<DEXDY>(f0);
    auto _dexdz    = Cabana::slice<DEXDZ>(f0);
    auto _d2exdydz = Cabana::slice<D2EXDYDZ>(f0);
    auto _ey       = Cabana::slice<EY>(f0);
    auto _deydz    = Cabana::slice<DEYDZ>(f0);
    auto _deydx    = Cabana::slice<DEYDX>(f0);
    auto _d2eydzdx = Cabana::slice<D2EYDZDX>(f0);
    auto _ez       = Cabana::slice<EZ>(f0);
    auto _dezdx    = Cabana::slice<DEZDX>(f0);
    auto _dezdy    = Cabana::slice<DEZDY>(f0);
    auto _d2ezdxdy = Cabana::slice<D2EZDXDY>(f0);
    auto _cbx      = Cabana::slice<CBX>(f0);
    auto _dcbxdx   = Cabana::slice<DCBXDX>(f0);
    auto _cby      = Cabana::slice<CBY>(f0);
    auto _dcbydy   = Cabana::slice<DCBYDY>(f0);
    auto _cbz      = Cabana::slice<CBZ>(f0);
    auto _dcbzdz   = Cabana::slice<DCBZDZ>(f0);

    auto _uncenter =
        //KOKKOS_LAMBDA( const int s ) {
        KOKKOS_LAMBDA( const int s, const int i ) {
            // Grab particle properties
            real_t dx = position_x.access(s,i);   // Load position
            real_t dy = position_y.access(s,i);   // Load position
            real_t dz = position_z.access(s,i);   // Load position

            int ii = cell.access(s,i);

            // Grab interpolator values (slices hoisted above)
            auto ex       = _ex(ii);
            auto dexdy    = _dexdy(ii);
            auto dexdz    = _dexdz(ii);
            auto d2exdydz = _d2exdydz(ii);
            auto ey       = _ey(ii);
            auto deydz    = _deydz(ii);
            auto deydx    = _deydx(ii);
            auto d2eydzdx = _d2eydzdx(ii);
            auto ez       = _ez(ii);
            auto dezdx    = _dezdx(ii);
            auto dezdy    = _dezdy(ii);
            auto d2ezdxdy = _d2ezdxdy(ii);
            auto cbx      = _cbx(ii);
            auto dcbxdx   = _dcbxdx(ii);
            auto cby      = _cby(ii);
            auto dcbydy   = _dcbydy(ii);
            auto cbz      = _cbz(ii);
            auto dcbzdz   = _dcbzdz(ii);

            // Calculate field values
            real_t hax = qdt_2mc*(( ex + dy*dexdy ) + dz*( dexdz + dy*d2exdydz ));
            real_t hay = qdt_2mc*(( ey + dz*deydz ) + dx*( deydx + dz*d2eydzdx ));
            real_t haz = qdt_2mc*(( ez + dx*dezdx ) + dy*( dezdy + dx*d2ezdxdy ));

            cbx = cbx + dx*dcbxdx;            // Interpolate B
            cby = cby + dy*dcbydy;
            cbz = cbz + dz*dcbzdz;

            // Load momentum
            real_t ux = velocity_x.access(s,i);   // Load velocity
            real_t uy = velocity_y.access(s,i);   // Load velocity
            real_t uz = velocity_z.access(s,i);   // Load velocity

            real_t v0 = qdt_4mc/(real_t)sqrt(one + (ux*ux + (uy*uy + uz*uz)));

            // Borris push
            // Boris - scalars
            real_t v1 = cbx*cbx + (cby*cby + cbz*cbz);
            real_t v2 = (v0*v0)*v1;
            real_t v3 = v0*(one+v2*(one_third+v2*two_fifteenths));
            real_t v4 = v3/(one+v1*(v3*v3));

            v4  += v4;

            v0   = ux + v3*( uy*cbz - uz*cby );      // Boris - uprime
            v1   = uy + v3*( uz*cbx - ux*cbz );
            v2   = uz + v3*( ux*cby - uy*cbx );

            ux  += v4*( v1*cbz - v2*cby );           // Boris - rotation
            uy  += v4*( v2*cbx - v0*cbz );
            uz  += v4*( v0*cby - v1*cbx );

            ux  += hax;                              // Half advance E
            uy  += hay;
            uz  += haz;

            // Store result
            velocity_x.access(s,i) = ux;
            velocity_y.access(s,i) = uy;
            velocity_z.access(s,i) = uz;

        };

    Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace>
        vec_policy( 0, particles.size() );
    Cabana::simd_parallel_for( vec_policy, _uncenter, "uncenter()" );
}

#endif // uncenter
