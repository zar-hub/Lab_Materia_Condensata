def thermocouple(tc_type,x,input_unit,output_unit=None) :
    """
    Converts thermocouple voltages into temperature, or vice versa.

    Accepts either single values or lists.

    Allowed thermocouple type tc_type are currently "E", "T", or "K"
    x is either the thermocouple voltage or temperature
    If input_unit = "C", "K", or "F", returns thermocouple emf.
    If input_unit = "V" or "mV", returns thermocouple temperature.
    Output_unit can be mV or V for emf, K, C, or F for temperature.

    Reference junction of thermocouple is at 0 C, i.e. ice water.
    Negative emfs correspond to temperatures below 0 C;
        positive emfs correspond to temperatures above 0 C. 

    Uses parameterizations found at http://srdata.nist.gov/its90/main/

    Copyright (c) 2011, 2012, 2014, 2015 University of Toronto
    Last Modification:  6 November 2015 by Eric Yeung (added Type E)
    Modified:           26 January 2014 by Michael Wainberg
                        3 March 2012 by David Bailey
    Original Version:   12 October 2011 by David Bailey
    Contact: David Bailey <dbailey@physics.utoronto.ca>
                        (http://www.physics.utoronto.ca/~dbailey)
    License: Released under the MIT License; the full terms are this license
                are appended to the end of this module, and are also available
                at http://www.opensource.org/licenses/mit-license.php.
    """

    import math

    # Initialize coefficients for calculating emf in millivolts for
    #   a given temperature in Celsius degrees
    c={}
    a={}
    # Initialize coefficients for calculating temperature in celsius degrees
    #   given a thermocouple emf in millivolts
    d={}
    # Initialize dictionaries for temperature and emf ranges for which
    #   conversions are valid
    t_range   = {}
    emf_range = {}

    # Type E Thermocouple values
    #   Uses polynomial parameterization found at
    #       http://srdata.nist.gov/its90/download/type_e.tab
    #           Retrieved 12 October 2011
    t_range["E"]   = [-270.0, 0.0, 1000.0, None]
    c["E"] = [
     [  # -270 to 0 degrees C
         0.000000000000E+00,
         0.586655087080E-01,
         0.454109771240E-04,
        -0.779980486860E-06,
        -0.258001608430E-07,
        -0.594525830570E-09,
        -0.932140586670E-11,
        -0.102876055340E-12,
        -0.803701236210E-15,
        -0.439794973910E-17,
        -0.164147763550E-19,
        -0.396736195160E-22,
        -0.558273287210E-25,
        -0.346578420130E-28
     ],[  # -0 to 1000 degrees C
         0.000000000000E+00,
         0.586655087100E-01,
         0.450322755820E-04,
         0.289084072120E-07,
        -0.330568966520E-09,
         0.650244032700E-12,
        -0.191974955040E-15,
        -0.125366004970E-17,
         0.214892175690E-20,
        -0.143880417820E-23,
         0.359608994810E-27
     ]
    ]
    a["E"] = [
         0.0,
         0.0,
         0.0
        ]

    emf_range["E"] = [-9.718, -8.825, 0.0, 76.373, None]
    d["E"] = [
     [ #    -9.718 to -8.825 mV, -250 to -200 degrees C
         20678,
         6889.2,
         762.19,
         28.285
     ],[  #   -8.825 to 0 mV, -200 to 0 degrees C, error range is -0.01 to 0.03C
         0.0000000E+00,
         1.6977288E+01,
        -4.3514970E-01,
        -1.5859697E-01,
        -9.2502871E-02,
        -2.6084314E-02,
        -4.1360199E-03,
        -3.4034030E-04,
        -1.1564890E-05,
         0.0000000E+00
     ],[ #   0 to 76.373mV, 0 to 1000 degrees C; error range is -0.02 to 0.02C
         0.0000000E+00,
         1.7057035E+01,
        -2.3301759E-01,
         6.5435585E-03,
        -7.3562749E-05,
        -1.7896001E-06,
         8.4036165E-08,
        -1.3735879E-09,
         1.0629823E-11,
        -3.2447087E-14
     ]
    ]


    # Type K Thermocouple values
    #   Uses polynomial parameterization found at
    #       http://srdata.nist.gov/its90/download/type_k.tab
    #           Retrieved 23 October 2011
    t_range["K"]   = [-270.0, 0.0, 1372.0, None]
    c["K"] = [
     [  # -270 to 0 degrees C
         0.000000000000E+00,
         0.394501280250E-01,
         0.236223735980E-04,
        -0.328589067840E-06,
        -0.499048287770E-08,
        -0.675090591730E-10,
        -0.574103274280E-12,
        -0.310888728940E-14,
        -0.104516093650E-16,
        -0.198892668780E-19,
        -0.163226974860E-22
     ],[  # -0 to 1372 degrees C
        -0.176004136860E-01,
         0.389212049750E-01,
         0.185587700320E-04,
        -0.994575928740E-07,
         0.318409457190E-09,
        -0.560728448890E-12,
         0.560750590590E-15,
        -0.320207200030E-18,
         0.971511471520E-22,
        -0.121047212750E-25
     ]
    ]

    a["K"] = [
         0.118597600000E+00,
        -0.118343200000E-03,
         0.126968600000E+03
        ]

    emf_range["K"] = [-5.891, 0.0, 20.644, 54.886]
    d["K"] = [
     [  #   -5.891 to 0 mV, -200 to 0 degrees C, error range is -0.02 to 0.04C
         0.0000000E+00,
         2.5173462E+01,
        -1.1662878E+00,
        -1.0833638E+00,
        -8.9773540E-01,
        -3.7342377E-01,
        -8.6632643E-02,
        -1.0450598E-02,
        -5.1920577E-04,
         0.0000000E+00
     ],[ #   0 to 20.644mV, 0 to 500 degrees C; error range is -0.05 to 0.04C
         0.000000E+00,
         2.508355E+01,
         7.860106E-02,
        -2.503131E-01,
         8.315270E-02,
        -1.228034E-02,
         9.804036E-04,
        -4.413030E-05,
         1.057734E-06,
        -1.052755E-08
     ],[ #   20.644mV to 54.886, 500 to 1372 C; error range -0.05 to 0.06C
        -1.318058E+02,
         4.830222E+01,
        -1.646031E+00,
         5.464731E-02,
        -9.650715E-04,
         8.802193E-06,
        -3.110810E-08,
         0.000000E+00,
         0.000000E+00,
         0.000000E+00
     ]
    ]

    # Type T Thermocouple values
    #   Uses polynomial parameterization found at
    #       http://srdata.nist.gov/its90/download/type_t.tab
    #           Retrieved 3 November 2015
    t_range["T"]   = [-270.0, 0.0, 400, None]
    c["T"] = [
     [  # -270 to 0 degrees C
         0.000000000000E+00,
         0.387481063640E-01,
         0.441944343470E-04,
         0.118443231050E-06,
         0.200329735540E-07,
         0.901380195590E-09,
         0.226511565930E-10,
         0.360711542050E-12,
         0.384939398830E-14,
         0.282135219250E-16,
         0.142515947790E-18,
         0.487686622860E-21,
         0.107955392700E-23,
         0.139450270620E-26,
         0.797951539270E-30
     ],[  # -0 to 400 degrees C
         0.000000000000E+00,
         0.387481063640E-01,
         0.332922278800E-04,
         0.206182434040E-06,
        -0.218822568460E-08,
         0.109968809280E-10,
        -0.308157587720E-13,
         0.454791352900E-16,
        -0.275129016730E-19
     ]
    ]
    a["T"] = [
         0.0,
         0.0,
         0.0
        ]

    emf_range["T"] = [-5.603, 0.0, 20.872, None]
    d["T"] = [
     [ #    -5.603 to -0 mV, -200 to 0 degrees C; error range is -0.02 to 0.04C
         0.0000000E+00,
         2.5949192E+01,
        -2.1316967E-01,
         7.9018692E-01,
         4.2527777E-01,
         1.3304473E-01,
         2.0241446E-02,
         1.2668171E-03
     ],[ #   0 to 20.872 mV, 0 to 400 degrees C; error range is -0.03 to 0.03C
         0.000000E+00,
         2.592800E+01,
        -7.602961E-01,
         4.637791E-02,
        -2.165394E-03,
         6.048144E-05,
        -7.293422E-07,
         0.000000E+00
     ]
    ]

    # Based on "input_unit", define conversion to be done
    #    The parameterizations assume temperature in C, emf in mV.
    if input_unit == "C" :
        temperature = x
        if output_unit == "mV" or output_unit == "V" or output_unit == None :
            if output_unit == None :
                output_unit = "mV"
            return_emf = True
        else :
            def wrong_output_unit() :
                # The output unit must be a known unit, and if the input is
                #     temperature, the output should be emf, and vice versa.
                print(("The output unit,'{0:s}', is either unrecognized or is "
                   + "the same quantity as the input unit").format(output_unit))
            wrong_output_unit()
            return None
    elif input_unit == "K" :
        temperature = x - 273.15
        if output_unit == "mV" or output_unit == "V" or output_unit == None :
            if output_unit == None :
                output_unit = "mV"
            return_emf = True
        else :
            wrong_output_unit()
            return None
    elif input_unit == "F" :
        temperature = (x - 32.0)*5./9.
        return_emf = True
    elif input_unit == "mV" :
        emf = x
        return_emf = False
    elif input_unit == "V" :
        emf = 1000.0*x
        return_emf = False
    else :
        print(("'{0:s}' is an unrecognized unit,"
              +" should be 'C', 'K', 'F', 'mV', or 'V'").format(input_unit))
        return None
    # Check if a valid thermocouple type
    invalid_thermocouple = True
    for k in emf_range.keys():
        if tc_type == k :
            invalid_thermocouple = False
    if invalid_thermocouple :
        return("'{0:s}' is not a recognized theromocouple type".format(tc_type))

    # Given temperature, return emf
    if return_emf :
        # Initialize temperature and emf (electromotive force, "voltage")
        try:
            # Create emf array if temperature is an array
            t= temperature
            v = len(temperature)*[0.0]
        except TypeError:
            t = [temperature]
            v = [0.0]
        for i in range(len(t)) :
            temperature_index = -1
            for j in range(len(t_range[tc_type])-1) :
                if t[i] >= t_range[tc_type][j] and t[i] < t_range[tc_type][j+1] :
                    temperature_index = j
                    break
            if temperature_index == -1 :
                return(("Temperature ({0:f} C) is outside allowed range for "+
                       "Type {1:s} thermocouple").format(t[i],tc_type))
            for j in range(len(c[tc_type][temperature_index])) :
                v[i] += c[tc_type][temperature_index][j]*t[i]**j
            v[i] += a[tc_type][0]*math.exp(
                a[tc_type][1]*(t[i]-a[tc_type][2])**2)
            if output_unit == "V" :
                v[i] = v[i]/1000.0
        if len(v) == 1 :
            # Return a value, not an array, if temperature was not an array
            return v[0]
        else :
            return v

    # Given emf, return temperature
    else :
        try:
            # Create emf array if temperature is an array
            v= emf
            t = len(emf)*[0.0]
        except TypeError:
            v = [emf]
            t = [0.0]
        for i in range(len(v)) :
            emf_index = -1
            for j in range(len(emf_range[tc_type])-1) :
                if v[i] >= emf_range[tc_type][j] and v[i] < emf_range[tc_type][j+1] :
                    emf_index = j
                    break
            if emf_index == -1 :
                return(("EMF ({0:f} mV) is outside allowed range for "+
                       "Type {1:s} thermocouple").format(v[i], tc_type))
            for j in range(len(d[tc_type][emf_index])) :
                t[i] += d[tc_type][emf_index][j]*v[i]**j
            # Note that temperature is returning in Kelvin, not Celsius
            if output_unit == "K" :
                t[i] += 273.15
            elif output_unit == "F" :
                t[i] = t[i]*9./5. + 32.0
        if len(t) == 1 :
            # Return a value, not an array, if emf was not an array
            return t[0]
        else :
            return t

"""
Full text of MIT License:

    Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""