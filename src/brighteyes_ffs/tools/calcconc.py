from .constants import constants


def calcconc(m, V, MM=368):
    """
    Calculate the molar concentration of a sample with mass m dissolved in
    volume V

    Parameters
    ----------
    m : float
        Mass of the solute [g].
    V : float
        Volume of the solution [L].
    MM : float, optional
        Molar mass of the solute [g/mole]
        Default set to 368 g/mole for Oregon Green. The default is 368.

    Returns
    -------
    c : float
        Concentration of the solution [mol/L].

    """
    
    Avogadro = constants('avogadro')
    
    # grams to moles of solute
    moles = m / MM
    
    # concentration [mole/L]
    c = moles / V
    
    # concentration [particles/L]
    c2 = c * Avogadro
    
    # concentration [particles/fL]
    c3 = c2 / 1e15
    
    # print results
    print('Concentration: {:.2e}'.format(c) + ' M = {:.2e}'.format(c2) + ' particles/L = {:.2e}'.format(c3) + ' particles/fL')
    
    return c
