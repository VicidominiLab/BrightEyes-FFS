def constants(c):
    """
    Physical constants

    Parameters
    ----------
    c : string
        name of the constant.

    Returns
    -------
    float
        constant value.

    """
    
    c = c.lower()
    if c == 'boltzmann':
        return 1.38064852e-23  # m^2 kg / (K s^2)
    elif c == 'planck':
        return 6.62607004e-35  # m^2 kg / s
    elif c == 'avogadro':
        return 6.02214076e23 # no units
    else:
        return 0
