def test_imports():
    import surface_renewal as sr
    from surface_renewal.methods import snyder, chen97, analysis
    from surface_renewal.preprocess import despike, rotation, stability