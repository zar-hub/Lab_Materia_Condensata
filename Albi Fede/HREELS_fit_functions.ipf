function pos_dsgn(w, x)
	wave w
	variable x
	return dsgn_MTHR(w, -x) 
end 


Function saturationExp(w,x) : FitFunc
	Wave w
	Variable x

	//CurveFitDialog/ These comments were created by the Curve Fitting dialog. Altering them will
	//CurveFitDialog/ make the function less convenient to work with in the Curve Fitting dialog.
	//CurveFitDialog/ Equation:
	//CurveFitDialog/ f(x) = y0 + A * (1 - Exp(-tau1 * (x - x0))) * Exp(-tau2 * x)
	//CurveFitDialog/ End of Equation
	//CurveFitDialog/ Independent Variables 1
	//CurveFitDialog/ x
	//CurveFitDialog/ Coefficients 5
	//CurveFitDialog/ w[0] = x0
	//CurveFitDialog/ w[1] = y0
	//CurveFitDialog/ w[2] = A
	//CurveFitDialog/ w[3] = tau1
	//CurveFitDialog/ w[4] = tau2

	return w[1] + w[2] * max((1 - Exp(-w[3] * (x - w[0]))), 0) * Exp(-w[4] * x)
End

Function saturationExpV2(w,x) : FitFunc
	Wave w
	Variable x

	//CurveFitDialog/ These comments were created by the Curve Fitting dialog. Altering them will
	//CurveFitDialog/ make the function less convenient to work with in the Curve Fitting dialog.
	//CurveFitDialog/ Equation:
	//CurveFitDialog/ f(x) = y0 + A * exp(t1 * (x-x0)) * exp(t2 * x )
	//CurveFitDialog/ End of Equation
	//CurveFitDialog/ Independent Variables 1
	//CurveFitDialog/ x
	//CurveFitDialog/ Coefficients 5
	//CurveFitDialog/ w[0] = base
	//CurveFitDialog/ w[1] = xhalf
	//CurveFitDialog/ w[2] = A
	//CurveFitDialog/ w[3] = rate
	//CurveFitDialog/ w[4] = t

	// The sigmoid models some sort of saturation
	make/free sigmoid_coeff = {0, 1, w[1], w[3]}

	return w[0] + w[2] * Exp(-w[4] * x) *  sigmoid(sigmoid_coeff, x)
End


Function sigmoid(w, x)
	wave w
	variable x
	
	return w[0]+w[1]/(1+exp(-(x-w[2])/w[3]))
end


function gaussnb(w, x)
	wave w
	variable x
	
	return w[0] * gauss(x, w[1], w[2])
end 

function const(w, x)
	wave w
	variable x
	
	return w[0]
end 

function doubleGauss(w, x)
	wave w
	variable x
	
	make/free/n=3 g1, g2
	g1 = w[p + 1]
	g2 = w[p + 4]
	
	return w[0] + gaussnb(g1, x) + gaussnb(g2, x)
end 

function lorAndGauss(w, x)
	wave w
	variable x	
	return w[0]/((x-w[1])^2+w[2]) + w[3]*exp(-((x-w[4])/w[5])^2)
end 
