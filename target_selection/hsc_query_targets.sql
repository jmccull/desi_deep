SELECT
    main.object_id,
    main.ra,
    main.dec,

    ------- flux and flux errors -------
    main2.g_psfflux_flux, main2.r_psfflux_flux, main2.i_psfflux_flux, main2.z_psfflux_flux, main2.y_psfflux_flux,
    main2.g_psfflux_fluxerr, main2.r_psfflux_fluxerr, main2.i_psfflux_fluxerr, main2.z_psfflux_fluxerr, main2.y_psfflux_fluxerr,
    main.g_cmodel_flux, main.r_cmodel_flux, main.i_cmodel_flux, main.z_cmodel_flux, main.y_cmodel_flux,
    main.g_cmodel_fluxerr, main.r_cmodel_fluxerr, main.i_cmodel_fluxerr, main.z_cmodel_fluxerr, main.y_cmodel_fluxerr,
------- Fiber flux and flux errors -------
main4.g_convolvedflux_2_15_flux as g_fiber_flux, main4.r_convolvedflux_2_15_flux as r_fiber_flux,
main4.i_convolvedflux_2_15_flux as i_fiber_flux, main4.z_convolvedflux_2_15_flux as z_fiber_flux, main4.y_convolvedflux_2_15_flux as y_fiber_flux,
main4.g_convolvedflux_2_15_fluxerr as g_fiber_fluxerr, main4.r_convolvedflux_2_15_fluxerr as r_fiber_fluxerr, main4.i_convolvedflux_2_15_fluxerr as i_fiber_fluxerr,
main4.z_convolvedflux_2_15_fluxerr as z_fiber_fluxerr, main4.y_convolvedflux_2_15_fluxerr as y_fiber_fluxerr,

main5.g_undeblended_convolvedflux_2_15_flux as g_fiber_tot_flux, main5.r_undeblended_convolvedflux_2_15_flux as r_fiber_tot_flux,
main5.i_undeblended_convolvedflux_2_15_flux as i_fiber_tot_flux, main5.z_undeblended_convolvedflux_2_15_flux as z_fiber_tot_flux, main5.y_undeblended_convolvedflux_2_15_flux as y_fiber_tot_flux,
main5.g_undeblended_convolvedflux_2_15_fluxerr as g_fiber_tot_fluxerr, main5.r_undeblended_convolvedflux_2_15_fluxerr as r_fiber_tot_fluxerr, main5.i_undeblended_convolvedflux_2_15_fluxerr as i_fiber_tot_fluxerr,
main5.z_undeblended_convolvedflux_2_15_fluxerr as z_fiber_tot_fluxerr, main5.y_undeblended_convolvedflux_2_15_fluxerr as y_fiber_tot_fluxerr,
    ------- extinction -------
    main.a_g, main.a_r, main.a_i, main.a_z, main.a_y,

    ------- fraction of flux in de Vaucouleur component -------
    main.g_cmodel_fracdev, main.r_cmodel_fracdev, main.i_cmodel_fracdev, main.z_cmodel_fracdev,
     ------- shape measurements -------
    main.g_extendedness_value, main.r_extendedness_value, main.i_extendedness_value, main.z_extendedness_va|lue,
    main.g_extendedness_flag, main.r_extendedness_flag, main.i_extendedness_flag, main.z_extendedness_flag,
    main2.i_sdssshape_shape11, main2.i_sdssshape_shape22, main2.i_sdssshape_shape12,
    main2.i_sdssshape_shape11err, main2.i_sdssshape_shape22err, main2.i_sdssshape_shape12err,
    ------- flags -------
    main2.g_sdsscentroid_flag, main2.r_sdsscentroid_flag, main2.i_sdsscentroid_flag, main2.z_sdsscentroid_flag, main2.y_sdsscentroid_flag, 
    main.g_pixelflags_edge, main.r_pixelflags_edge, main.i_pixelflags_edge, main.z_pixelflags_edge, main.y_pixelflags_edge, 
    main.g_pixelflags_interpolatedcenter, main.r_pixelflags_interpolatedcenter, main.i_pixelflags_interpolatedcenter, main.z_pixelflags_interpolatedcenter, main.y_pixelflags_interpolatedcenter, 
    main.g_pixelflags_saturatedcenter, main.r_pixelflags_saturatedcenter, main.i_pixelflags_saturatedcenter, main.z_pixelflags_saturatedcenter, main.y_pixelflags_saturatedcenter, 
    main.g_pixelflags_crcenter, main.r_pixelflags_crcenter, main.i_pixelflags_crcenter, main.z_pixelflags_crcenter, main.y_pixelflags_crcenter, 
    main.g_pixelflags_bad, main.r_pixelflags_bad, main.i_pixelflags_bad, main.z_pixelflags_bad, main.y_pixelflags_bad, 
    main.g_cmodel_flag, main.r_cmodel_flag, main.i_cmodel_flag, main.z_cmodel_flag, main.y_cmodel_flag,
    -----Star masks -----
    mask.g_mask_brightstar_any, mask.g_mask_brightstar_halo, mask.g_mask_brightstar_dip,
    mask.g_mask_brightstar_ghost, mask.g_mask_brightstar_blooming, mask.g_mask_brightstar_ghost12, mask.g_mask_brightstar_ghost15,
    mask.r_mask_brightstar_any, mask.r_mask_brightstar_halo, mask.r_mask_brightstar_dip,
    mask.r_mask_brightstar_ghost, mask.r_mask_brightstar_blooming, mask.r_mask_brightstar_ghost12, mask.r_mask_brightstar_ghost15,
    mask.i_mask_brightstar_any, mask.i_mask_brightstar_halo, mask.i_mask_brightstar_dip,
    mask.i_mask_brightstar_ghost, mask.i_mask_brightstar_blooming, mask.i_mask_brightstar_ghost12, mask.i_mask_brightstar_ghost15,
    mask.z_mask_brightstar_any, mask.z_mask_brightstar_halo, mask.z_mask_brightstar_dip,
    mask.z_mask_brightstar_ghost, mask.z_mask_brightstar_blooming, mask.z_mask_brightstar_ghost12, mask.z_mask_brightstar_ghost15,
    mask.y_mask_brightstar_any, mask.y_mask_brightstar_halo, mask.y_mask_brightstar_dip,
    mask.y_mask_brightstar_ghost, mask.y_mask_brightstar_blooming, mask.y_mask_brightstar_ghost12, mask.y_mask_brightstar_ghost15

FROM
    pdr3_wide.forced main
    LEFT JOIN pdr3_wide.forced2 main2 USING (object_id)
    LEFT JOIN pdr3_wide.forced4 main4 USING (object_id)
    LEFT JOIN pdr3_wide.forced5 main5 USING (object_id)
	LEFT JOIN pdr3_wide.masks mask USING (object_id)

WHERE
    isprimary
    AND i_cmodel_mag<24.8
    AND boxSearch(coord, 33.5, 37.5, -7.0, -3.0) ---XMM
    ---AND boxSearch(coord, 148, 152, 4, 0) ---COSMOS
    ---AND boxSearch(coord, 213, 217, 54, 51.75) ---DEEP23
    ---AND boxSearch(coord, 243, 249, 41, 45) ---HERCULES
