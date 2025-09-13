// === UTILITY FUNCTIONS ===
function normalize(src)
	wave src
	string dup_name = "Norm_" + nameofwave(src)
	variable m = wavemax(src)
	duplicate/O src, $dup_name
	
	wave dup = $dup_name
	dup = dup / m
end

function	find_max(src) 
	// returns the position
	// of maximum intensity
	wave src
	variable m = wavemax(src)
	findvalue/v =(m) src
	return V_value
end

function displace(src, dx)
	wave src
	variable dx
	variable offset = dimoffset(src, 0)
	variable delta = dimdelta(src, 0)
	offset = offset + dx
	setscale/p x, offset, delta, "", src
end

function/wave new_append(src, s)
	wave src
	string s
	
	string name = nameofwave(src)
	string new = name + s
	duplicate/o src, $new 
	wave new_wave = $new 
	new_wave = Nan
	
	return $new
end

function/wave new_append_coeff(src, coeff, s)
	wave src, coeff
	string s
	
	string name = nameofwave(src)
	string new = name + s
	duplicate/o coeff, $new 
	wave new_wave = $new 
	new_wave = Nan
	
	return $new
end

function/wave copy_append(src, s)
	wave src
	string s
	
	string name = nameofwave(src)
	string new = name + s
	duplicate/o src, $new 
	return $new
end

function/wave copy_append_coeff(src, coeff, s)
	wave src, coeff
	string s
	
	string name = nameofwave(src)
	string new = name + s
	duplicate/o coeff, $new 	
	return $new
end

// === PEAK ANALYSIS ===
// THE FOLLOWING FUNCTIONS ARE SUPPOSED TO USE
// IN ORDER
function calibrate_peak_pos(src, pos)
	wave src 
	variable pos
	wave peak
	
	variable peak_i = find_max(src)
	print "Peak at index " + num2str(peak_i)
	
	// do a quick fit to estimate 
	// the center
	CurveFit/M=2/W=0 gauss, src[peak_i - 3,peak_i + 3]/D
	
	// save the peak position
	wave coeff = $"W_coef"
	variable peak_pos = coeff[2]
	print "Peak at position " + num2str(peak_pos)
	
	// displace the peak to the position
	displace(src, - peak_pos)
	print "Calibrated peak position"
end

function get_sym_peak(src, x)
	// generates a symmetric peak by
	// mirroring the wave across the 
	// vertical axis passing trough x
	wave src
	variable x
	wave mirror = new_append(src, "_MIRROR")
	reverse/p src/d= mirror
	
	// displace by the correct amount
	variable n = numpnts(src)
	variable start = pnt2x(src, 0)
	variable finish = pnt2x(src, n - 1)
	
	variable a = x - start
	variable b = finish - x
	variable delta = b - a
	displace(mirror, - delta)
	
	// get the peak
	string srcName = nameofwave(src)
	string mirrorName = nameofwave(mirror)
	string peakName = srcName + "_SYM_PEAK"
	
	// find the right points to assign
	variable delta_pos = x2pnt(src, x)
	make/o/n=(2 * delta_pos) $peakName
	wave peak = $peakName
	copyscales/p src, peak

	// combine the two using a command
	string cmd = peakName+"=min("+srcName+"(x),"+mirrorName+"(x))"
	execute cmd
	
	display peak
end

function fit_zlp(initial_coeff, src)
	// fit the peak using a combination of lorentian and gaussiam
	// no background
	wave src
	wave initial_coeff
	wave res = new_append(src, "_FIT")
	wave coeff = new_append_coeff(src, initial_coeff, "_COEFF")
	Funcfit/TBOX=768 lorandGauss, coeff, src/d=res
end

function new_zlp(initial_coeff, src)
	// create zero loss peak for source
	wave src
	wave initial_coeff
	
	string srcName = nameofwave(src)
	string zlpName = srcName + "_ZLP"
	string coeffName = srcName + "_ZLP_COEFF"
	string inelasticName = srcName + "_INELASTIC"
	
	// make the waves
	duplicate/o initial_coeff, $coeffName
	duplicate/o src, $zlpName, $inelasticName
	
	wave coeff = $coeffName
	wave zlp = $zlpName
	wave inelastic = $inelasticName
	
	// ZLP FIRST FIT
	FuncFit pos_dsgn, coeff, src[6,16] 
	print coeff
	zlp = pos_dsgn(coeff, -x)
	
	inelastic = src - zlp
end

// === BACKGROUND FOR INELASTIC PEAK ===
function fit_expBg(initial_coeff, src)
	// first do a manual fit, than refine 
	// with this function

	wave src
	wave initial_coeff
	
	wave inelastic = $nameOfWave(src) + "_INELASTIC"
	wave bg = new_append(inelastic, "_BG")
	wave peaks = new_append(inelastic, "_PEAKS")
	wave mask = copy_append(inelastic, "_BG_MASK")
	wave coeff = copy_append_coeff(inelastic, initial_coeff, "_BG_COEFF")
	
	
	// start by fitting the exponential tail
	// after 1eV
	variable i, j
	i = x2pnt(inelastic, 0.3)
	j = numpnts(inelastic)

	// lock x0 and tau1
	FuncFit/x=1/h="10010" saturationExp coeff inelastic[i, j - 1]
	
	// fit the initial slope
	// set the mask
	i = x2pnt(inelastic, 0.1)
	j = x2pnt(inelastic, 0.7) 
	mask[i,j] = 0
	FuncFit/x=1/h="01101" saturationExp coeff inelastic/m=mask
	
	// save the results
	bg = saturationExp(coeff, x)
	peaks = inelastic - bg
end

function fit_bgIPRPTCDI(initial_coeff, src)
	// first do a manual fit, than refine 
	// with this function

	wave src
	wave initial_coeff
	
	wave inelastic = $nameOfWave(src) + "_INELASTIC"
	wave bg = new_append(inelastic, "_BG")
	wave peaks = new_append(inelastic, "_PEAKS")
	wave mask = copy_append(inelastic, "_BG_MASK")
	wave coeff = copy_append_coeff(inelastic, initial_coeff, "_BG_COEFF")
	
	
	// start by fitting the exponential tail
	// after 1eV
	variable i, j
	i = x2pnt(inelastic, 0.5)
	j = numpnts(inelastic)

	// lock x0 and tau1
	FuncFit/x=1/h="10010" saturationExp coeff inelastic[i, j - 1]
	
	// fit the initial slope
	// set the mask
	i = x2pnt(inelastic, 0.1)
	j = x2pnt(inelastic, 0.7) 
	mask[i,j] = 0
	j = x2pnt(inelastic, 0) 
	mask[0,j] = 0
	FuncFit/x=1/h="11011" saturationExp coeff inelastic/m=mask
	
	// save the results
	bg = saturationExp(coeff, x)
	peaks = inelastic - bg
end

function calibrate_bg(src, delta)
	wave src
	variable delta
	wave initial_coeff
	
	wave inelastic = $nameOfWave(src) + "_INELASTIC"
	wave bg = new_append(inelastic, "_BG")
	wave peaks = new_append(inelastic, "_PEAKS")
	wave coeff = copy_append_coeff(inelastic, initial_coeff, "_BG_COEFF")
	duplicate/free src res
	
	// Tune down BG until every point of the fit after
	// 0.1 eV is over the bg fit.
	// his way its nice when seen
	// in log scale.
	variable i, j, m
	m = find_max(inelastic)
	
	for(i=0; i<100; i+=1)
	
		// get the residual
		bg = saturationExp(coeff, x)
		peaks = inelastic - bg
		
		// truncate all the 
		// points before the max
		res = peaks
		res[0, m] = delta +1
	
		if (wavemin(res) >= -delta)
			break
		endif
		
		// remove 1% if bg is too high.
		// This keeps the same slope in 
		// log scale but turns down the 
		// intercept
		coeff[2] = coeff[2] * 0.99
	endfor
	print "Calibrated after " + num2str(i) + " iterations"
end
	
function get_peaks(src)
	wave src

	string srcName = nameofwave(src)
	string peaksName =  srcName + "_PEAKS"
	
	wave base = $(srcName + "_BASE")
	
	// make the peaks
	duplicate/o $srcName, $peaksName
	
	if(waveexists(base) == 0)
		print "Error: no base wave found!"
		return -1
	endif
	
	wave peaks = $peaksName
	peaks = src - base
end

function fit_base(src)
	wave src
	string srcName = nameofwave(src)
	string zlpName = srcName + "_ZLP"
	string bgName = srcName + "_BG"
	string baseName = srcName + "_BASE"	
	
	// Add ZLP, BG and BASE if they do not exist
	if(waveexists($srcName + "_ZLP_COEFF") == 0)
		print "Init ZLP"
		make/free c = {0.01, 0.05, 0.03, 15000, 0}
		new_zlp(c, src)
	endif
	if(waveexists($ srcName + "_BG_COEFF") == 0)
		print "Init BG"
		make/free c = {0, 0.01, 360, 0.01, 2.23}
		fit_expBg(c, src)
	endif
	if(waveexists($ srcName + "_BASE") == 0)
		print "Init BASE"
		duplicate/o src, $ srcName + "_BASE"
	endif
	if(waveexists($ srcName + "_RES") == 0)
		print "Init RES"
		duplicate/o src, $ srcName + "_RES"
	endif

	wave zlp = $ srcName + "_ZLP"
	wave bg = $ srcName + "_BG"
	wave base = $ srcName + "_BASE"
	wave res = $ srcName + "_RES"
	wave inelastic = $srcName + "_INELASTIC"
	duplicate/free res, tmp
	
	// TODO: automatic weights settings
	print "Adding weights"
	duplicate/o src weights
	weights = 1
	weights[24,65] = 0
	
	FuncFit/TBOX=768 {{dsgnnb_MTHR, d}, {saturationExpV2, PTCDI_17_12_2024_R1_BG_COEFF, HOLD="01000"}} src/M=weights
	// save the fit 
	zlp = dsgnnb_MTHR($"PTCDI_17_12_2024_R1_ZLP_COEFF", x)
	bg = saturationExpV2($"PTCDI_17_12_2024_R1_BG_COEFF", x)
	inelastic = src - zlp
	
	
	// Tune down BG until every point after the zlp 
	// is over the bg : this way its nice when seen
	// in log scale.
	variable i, N
	N = numpnts(base)
	
	for(i=0; i<20; i+=1)
		base = zlp + bg
		res = src - base
		
		// truncate all the non interesting points
		tmp = res
		tmp[0,25] = 0
		
		if (wavemin(tmp) >= 0)
			print "Calibrated BG"
			break
		endif
		
		// remove 1% if bg is too high.
		// This keeps the same slope in 
		// log scale but turns down the 
		// intercept
		bg = bg * 0.99
	endfor
	print "Attenuated BG for " + num2str(i) + " times" 
	
	string gname =  srcName + "_BaseFIT"
	display/n = $(gname)
	appendToGraph src, zlp, bg, base, weights
end

Function TransAx_eVtoCm_1(w, val)
    Wave/Z w            // a parameter wave, if desired. Argument must be present for FindRoots.
    Variable val
    
    return val*8065.61042
end

Function/S TickMeV(val)
	variable val
   String TickMeV
   TickMeV = num2str(val * 1000)  // Convert eV to meV
End
