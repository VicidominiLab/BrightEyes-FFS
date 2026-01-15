# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 16:30:32 2025

@author: eslenders
"""


def get_corr_input(corrtype, det):
    """
    Make input for correlation calculation by using predefined keywords.
    The output of this function can be used in atimes_file_2_corr()
    which takes list_of_g, list_of_g_out, and averaging as input parameters

    In other words, the function atimes_file_2_corr needs several input
    parameters describing which correlations to calculate, over which to
    average, and what their output names should be. Instead of manually entering
    these parameters, you can use this function, which contains the most used
    correlation and detector types.

    Parameters
    ----------
    corrtype : str
        Describe which correlation to calculate. Either
        'all autocorrelations'
        'spot-variation fcs'
        'spot-variation fcs - extended'
        'pair-correlation fcs'
        'all cross-correlations'
        
    det : str
        Describe which detector is used. Either
        'square 5x5'
        'pda-23'
        'airyscan 32'
        
    Returns
    -------
    list_of_g : list of str
        Describes all correlations that need to be calculated.
    list_of_g_out : list of str
        Gives user-friendly names to these correlations.
    averaging : list of str
        Describes which correlations need to be averaged.

    """
    
    corrtype = corrtype.lower()
    det = det.lower()
    
    det_GI5 = ['square 5x5', 'nsparc', '5x5']
    det_PDA23 = ['pda-23', 'luminosa', 'pda23']
    det_Airyscan = ['airyscan', 'airyscan 32']
    
    # -------------------- All autocorrelations --------------------
    if corrtype in ['all autocorrelations', 'fcs']:
        if det in det_GI5:
            N = 25
            list_of_g_out = ['det' + str(i) + 'x' + str(i) for i in range(N)]
            list_of_g = [i for i in range(N)]
            averaging = None
            return list_of_g, list_of_g_out, averaging
            
        if det in det_PDA23:
            N = 23
            list_of_g_out = ['det' + str(i+9) + 'x' + str(i+9) for i in range(N)]
            list_of_g = [i+9 for i in range(N)]
            averaging = None
            return list_of_g, list_of_g_out, averaging
        
        if det in det_Airyscan:
            N = 32
            list_of_g_out = ['det' + str(i) + 'x' + str(i) for i in range(N)]
            list_of_g = [i for i in range(N)]
            averaging = None
            return list_of_g, list_of_g_out, averaging
    
    
    # -------------------- Spot-variation FCS --------------------
    if corrtype in ['spot-variation fcs', 'svfcs']:
        if det in det_GI5:
            list_of_g_out = ['central', 'sum3', 'sum5']
            list_of_g = ['central', 'sum3', 'sum5']
            averaging = None
            return list_of_g, list_of_g_out, averaging
            
        if det in det_PDA23:
            list_of_g_out = ['central', 'sum7', 'sum19']
            list_of_g = ['picentral', 'piring1', 'piring2']
            averaging = None
            return list_of_g, list_of_g_out, averaging
        
        if det in det_Airyscan:
            list_of_g_out = ['central', 'ring1', 'ring2', 'ring3']
            list_of_g = ['airycentral', 'airyring1',  "airyring2", "airyring3"]
            averaging = None
            return list_of_g, list_of_g_out, averaging
    
    
    # -------------------- Spot-variation FCS extended --------------------
    if corrtype in ['spot-variation fcs - extended', 'svfcs-ext']:
        if det in det_GI5:
            list_of_g_out = ['sum1', 'sum5', 'sum9', 'sum13', 'sum21', 'sum25']
            list_of_g = ['central', 'C7+11+12+13+17', 'sum3', 'C2+6+7+8+10+11+12+13+14+16+17+18+22', 'C2+6+7+8+10+11+12+13+14+16+17+18+22+1+3+5+9+15+19+21+23', 'sum5']
            averaging = None
            return list_of_g, list_of_g_out, averaging
            
        if det in det_PDA23:
            list_of_g_out = ['central', 'sum7', 'sum19', 'sum23']
            list_of_g = ['picentral', 'piring1', 'piring2', 'piring3']
            averaging = None
            return list_of_g, list_of_g_out, averaging
        
        if det in det_Airyscan:
            list_of_g_out = ['central', 'ring1', 'ring2', 'ring3']
            list_of_g = ['airycentral', 'airyring1',  "airyring2", "airyring3"]
            averaging = None
            return list_of_g, list_of_g_out, averaging
        
        
    # -------------------- Pair-correlation FCS --------------------
    if corrtype in ['pair-correlation fcs']:
        if det in det_GI5:
            list_of_g_out = ['s0', 's1', 's2', 's4', 's5', 's8']
            list_of_g = ['crossAll']
            averaging = ['12x12', '12x7+12x11+12x13+12x17', '12x6+12x8+12x16+12x18', '12x2+12x10+12x14+12x22', '12x1+12x3+12x5+12x9+12x15+12x19+12x21+12x23', '12x0+12x4+12x20+12x24']
            return list_of_g, list_of_g_out, averaging
            
        if det in det_PDA23:
            list_of_g_out = ['s0', 's1', 's2']
            list_of_g = ['x2020', 'x2015', 'x2016', 'x2019', 'x2021', 'x2024', 'x2025', 'x2012', 'x2010', 'x2018', 'x2022', 'x2028', 'x2030']
            averaging = ['20x20', '20x15+20x16+20x19+20x21+20x24+20x25', '20x12+20x10+20x18+20x22+20x28+20x30']
            return list_of_g, list_of_g_out, averaging
        
        if det in det_Airyscan:
            list_of_g_out = ['s1', 's2', 's3', 's4']
            list_of_g = ['crossAll']
            averaging = ['0x1+0x2+0x3+0x4+0x5+0x6', '0x7+0x9+0x11+0x13+0x15+0x17', '0x8+0x10+0x12+0x14+0x16+0x18', '0x23+0x24+0x25+0x26+0x27+0x28+0x29+0x30+0x19+0x20+0x21+0x22']
            return list_of_g, list_of_g_out, averaging
        
    
    # -------------------- All cross-correlations with symmetry averaging --------------------
    if corrtype in ['all cross-correlations', 'cross-correlation ffs', 'xcorr', 'xcorrs', 'crossall']:
        if det in det_GI5:
            list_of_g_out = ['V-4_H0', 'V-3_H-1', 'V-3_H0', 'V-3_H1', 'V-2_H-2', 'V-2_H-1', 'V-2_H0', 'V-2_H1', 'V-2_H2', 'V-1_H-3', 'V-1_H-2', 'V-1_H-1', 'V-1_H0', 'V-1_H1',
            'V-1_H2', 'V-1_H3', 'V0_H-4', 'V0_H-3', 'V0_H-2', 'V0_H-1', 'V0_H0', 'V0_H1', 'V0_H2', 'V0_H3', 'V0_H4', 'V1_H-3', 'V1_H-2', 'V1_H-1', 'V1_H0', 'V1_H1', 'V1_H2',
            'V1_H3', 'V2_H-2', 'V2_H-1', 'V2_H0', 'V2_H1', 'V2_H2', 'V3_H-1', 'V3_H0', 'V3_H1', 'V4_H0']
            list_of_g = ['crossAll']
            averaging = ['22x2', '18x2+22x6', '17x2+22x7', '16x2+22x8', '14x2+18x6+22x10', '13x2+17x6+18x7+22x11', '12x2+16x6+17x7+18x8+22x12', '11x2+16x7+17x8+22x13',
                         '10x2+16x8+22x14', '14x6+18x10', '13x6+14x7+17x10+18x11', '8x2+12x6+13x7+14x8+16x10+17x11+18x12+22x16', '7x2+11x6+12x7+13x8+16x11+17x12+18x13+22x17',
                         '6x2+10x6+11x7+12x8+16x12+17x13+18x14+22x18', '10x7+11x8+16x13+17x14', '10x8+16x14', '14x10', '13x10+14x11', '8x6+12x10+13x11+14x12+18x16',
                         '7x6+8x7+11x10+12x11+13x12+14x13+17x16+18x17', '2x2+6x6+7x7+8x8+10x10+11x11+12x12+13x13+14x14+16x16+17x17+18x18+22x22', '6x7+7x8+10x11+11x12+12x13+13x14+16x17+17x18',
                         '6x8+10x12+11x13+12x14+16x18', '10x13+11x14', '10x14', '8x10+14x16', '7x10+8x11+13x16+14x17', '2x6+6x10+7x11+8x12+12x16+13x17+14x18+18x22',
                         '2x7+6x11+7x12+8x13+11x16+12x17+13x18+17x22', '2x8+6x12+7x13+8x14+10x16+11x17+12x18+16x22', '6x13+7x14+10x17+11x18', '6x14+10x18', '2x10+8x16+14x22',
                         '2x11+7x16+8x17+13x22', '2x12+6x16+7x17+8x18+12x22', '2x13+6x17+7x18+11x22', '2x14+6x18+10x22', '2x16+8x22', '2x17+7x22', '2x18+6x22', '2x22']
            return list_of_g, list_of_g_out, averaging
            
        if det in det_PDA23:
            list_of_g_out = []
            for angle in [0, 60, 120, 180, 240, 300]:
                for dist in [1, 2]:
                    list_of_g_out.append('Angle' + str(angle) + '_' + str(dist))
            list_of_g = ['crossAll']
            averaging = ['12x11+11x10+17x16+16x15+15x14+22x21+21x20+20x19+19x18+26x25+25x24+24x23+30x29+29x28', '12x10+17x15+16x14+22x20+21x19+20x18+26x24+25x23+30x28',
                       '10x14+14x18+11x15+15x19+19x23+12x16+16x20+20x24+24x28+17x21+21x25+25x29+22x26+26x30', '10x18+11x19+15x23+12x20+16x24+20x28+17x25+21x29+22x30',
                       '12x17+17x22+11x16+16x21+21x26+10x15+15x20+20x25+25x30+14x19+19x24+24x29+18x23+23x28', '12x22+11x21+16x26+10x20+15x25+20x30+14x24+19x29+18x28',
                       '11x12+10x11+16x17+15x16+14x15+21x22+20x21+19x20+18x19+25x26+24x25+23x24+29x30+28x29', '10x12+15x17+14x16+20x22+19x21+18x20+24x26+23x25+28x30',
                       '14x10+18x14+15x11+19x15+23x19+16x12+20x16+24x20+28x24+21x17+25x21+29x25+26x22+30x26', '18x10+19x11+23x15+20x12+24x16+28x20+25x17+29x21+30x22',
                       '17x12+22x17+16x11+21x16+26x21+15x10+20x15+25x20+30x25+19x14+24x19+29x24+23x18+28x23', '22x12+21x11+26x16+20x10+25x15+30x20+24x14+29x19+28x18']
            return list_of_g, list_of_g_out, averaging
        
        if det in det_Airyscan:
            list_of_g_out = []
            for angle in [0, 60, 120, 180, 240, 300]:
                for dist in [1, 2]:
                    list_of_g_out.append('Angle' + str(angle) + '_' + str(dist))
            list_of_g = ['crossAll']
            averaging = ['10x9+9x8+11x2+2x1+1x7+12x3+3x0+0x6+6x18+13x4+4x5+5x17+14x15+15x16', '0x18+3x6+12x0+2x7+11x1+10x8+4x17+13x5+14x16',
                         '8x7+7x18+9x1+1x6+6x17+10x2+2x0+0x5+5x16+11x3+3x4+4x15+12x13+13x14', '0x16+2x5+10x0+1x17+9x6+8x18+3x15+11x4+12x14',
                         '10x11+11x12+9x2+2x3+3x13+8x1+1x0+0x4+4x14+7x6+6x5+5x15+18x17+17x16', '0x14+1x4+8x0+2x13+9x3+10x12+6x15+7x5+18x16',
                         '8x9+9x10+7x1+1x2+2x11+18x6+6x0+0x3+3x12+17x5+5x4+4x13+16x15+15x14', '0x12+6x3+18x0+17x4+5x13+16x14+7x2+1x11+8x10',
                         '18x7+7x8+17x6+6x1+1x9+16x5+5x0+0x2+2x10+15x4+4x3+3x11+14x13+13x12', '0x10+5x2+16x0+6x9+17x1+18x8+15x3+4x11+14x12',
                         '12x11+11x10+13x3+3x2+2x9+14x4+4x0+0x1+1x8+15x5+5x6+6x7+16x17+17x18', '0x8+4x1+14x0+3x9+13x2+12x10+5x7+15x6+16x18']
            return list_of_g, list_of_g_out, averaging
        
        
    # -------------------- All cross-correlations without symmetry averaging --------------------
    if corrtype in ['crossall-ext']:
        if det in det_GI5:
            list_of_g_out = None
            list_of_g = ['crossAll']
            averaging = None
            return list_of_g, list_of_g_out, averaging
            
        if det in det_PDA23:
            list_of_g_out = None
            list_of_g = ['crossAll']
            averaging = None
            return list_of_g, list_of_g_out, averaging
        
        if det in det_Airyscan:
            list_of_g_out = None
            list_of_g = ['crossAll']
            averaging = None
            return list_of_g, list_of_g_out, averaging
        
        
    return None, None, None
            