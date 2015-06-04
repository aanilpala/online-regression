package regression.online.learner;

import regression.online.model.Prediction;
import regression.online.util.MatrixOp;
import regression.online.util.NLInputMapper;

public class BayesianPredictiveWindowed extends WindowRegressor {
	
	double[][] mul1;
	double[][] mul2;   // column matrix
	
	double a; //measurement_precision;
	double b; //weight_precision;
	
	public BayesianPredictiveWindowed(int input_width, boolean map2fs, double signal_stddev, double weight_stddev) {
		
		super(map2fs, input_width);
		
		name = "BayesianPredictiveWindowed" + (map2fs ? "_MAPPED" : "");
		
		a = 1/(signal_stddev*signal_stddev);
		b = 1/(weight_stddev*weight_stddev);
	
		mul1 = new double[feature_count][feature_count];
		mul2 = new double[feature_count][1];
		
		for(int ctr = 0; ctr < feature_count; ctr++) {
			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
				if(ctr == ctr2) mul1[ctr][ctr2] = b/a;
				else mul1[ctr][ctr2] = 0;
			}
		}
		
		for(int ctr = 0; ctr < feature_count; ctr++) {
			for(int ctr3 = 0; ctr3 < 3; ctr3++) {
				mul2[ctr][0] = 0;
			}
		}
		
	}
	
	public Prediction predict(double[][] dp) throws Exception {
		
		if(map2fs) dp = nlinmap.map(dp);
		
		double pp = MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(dp), mul1), mul2)[0][0];
		
		double pred_variance = MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(dp), MatrixOp.scalarmult(mul1, 1/a)), dp)[0][0]; 
		
		return new Prediction(pp, pp + 1.96*(Math.sqrt(pred_variance)), pp - 1.96*(Math.sqrt(pred_variance)));
		
	}
	
	public void update(double[][] dp, double y, Prediction prediction) throws Exception {
		
		if(map2fs) dp = nlinmap.map(dp);
				
		// update v
		
		mul2 = MatrixOp.mat_add(mul2, MatrixOp.scalarmult(dp, y));
		
		// update A^-1
		
		double denom = 1 + MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(dp), mul1), dp)[0][0];
		
		double[][] nom1 = MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(mul1, dp), MatrixOp.transpose(dp)),mul1);
		
		// compute b
		
		mul1 = MatrixOp.mat_add(mul1, MatrixOp.scalarmult(nom1, -1/denom));
		
		if(slide) {
			
			// update mul2
			
			mul2 = MatrixOp.mat_add(mul2, MatrixOp.scalarmult(dp_window[w_start], -1*responses[w_start][0]));
			
			// rank-1 downdate mul1 (inverse of X'X^-1)
			
			denom = -1 + MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(dp_window[w_start]), mul1), dp_window[w_start])[0][0];
			
			nom1 = MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(mul1, dp_window[w_start]), MatrixOp.transpose(dp_window[w_start])),mul1);
			
			// compute params
			
			mul1 = MatrixOp.mat_add(mul1, MatrixOp.scalarmult(nom1, -1/denom));
						
			// window is full
			// replace the element w_start points with the new element
			// increment w_start and w_end by 1
			
			for(int ctr = 0; ctr < feature_count; ctr++) {
				dp_window[w_end][ctr][0] = dp[ctr][0];
			}
			
			responses[w_end][0] = y;
			
			w_start = (w_start + 1) % w_size;
			w_end = (w_end + 1) % w_size;
		
		}
		else {
			
			// window is not full. accumulate dp's
			
			for(int ctr = 0; ctr < feature_count; ctr++) {
				dp_window[w_end][ctr][0] = dp[ctr][0];
			}
			
			responses[w_end][0] = y;
			
			w_end = (w_end + 1) % w_size;
			
			if(w_end == w_start) slide = true;
		}
		
	}
	
}
