package regression.online.learner;

import regression.online.model.Prediction;
import regression.online.util.MatrixOp;
import regression.online.util.NLInputMapper;

public class BayesianMLEWindowed extends WindowRegressor {
	
	double[][] mul1;
	double[][] mul2;	// column matrices
	double[][] params; // column matrices
	
	double running_residual_variance;
	
	public BayesianMLEWindowed(int input_width, boolean map2fs) {
		
		super(map2fs, input_width);
		
		mul1 = new double[feature_count][feature_count];
		mul2 = new double[feature_count][1];
		params = new double[feature_count][1];
		
		for(int ctr = 0; ctr < feature_count; ctr++) {
			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
				if(ctr == ctr2) mul1[ctr][ctr2] = 10000;
				else mul1[ctr][ctr2] = 0;
			}
		}
		
		for(int ctr = 0; ctr < feature_count; ctr++) {
			mul2[ctr][0] = 0;
			params[ctr][0] = 0;
		}
		
		running_residual_variance = 0;
		
	}
	
	public Prediction predict(double[][] dp) throws Exception {
		
		if(map2fs) dp = nlinmap.map(dp);
		
		double pp = MatrixOp.mult(MatrixOp.transpose(dp), params)[0][0];
		
		double predictive_deviation = Math.sqrt(running_residual_variance*MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(dp), mul1), dp)[0][0] + running_residual_variance);
		
		return new Prediction(pp, predictive_deviation);
		
	}
	
	public void update(double[][] dp, double y, Prediction prediction) throws Exception {
		
		int index = getIndexForDp(dp);
		
		if(index != -1) {
			// reject the update
			// avg the response for the duplicate point
			
			responses[index][0] = (y + responses[index][0])/2.0;
			return;
		}
		
		if(map2fs) dp = nlinmap.map(dp);
		
		// update mul2
		
		mul2 = MatrixOp.mat_add(mul2, MatrixOp.scalarmult(dp, y));
		
		// rank-1 update mul1 (inverse of X'X^-1)
		
		double denom = 1 + MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(dp), mul1), dp)[0][0];
		
		double[][] nom1 = MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(mul1, dp), MatrixOp.transpose(dp)),mul1);
		
		// compute params
		
		mul1 = MatrixOp.mat_add(mul1, MatrixOp.scalarmult(nom1, -1/denom));
		
		params = MatrixOp.mult(mul1, mul2);
		
		// slide the window
		
		if(slide) {
			
			// update mul2
			
			mul2 = MatrixOp.mat_add(mul2, MatrixOp.scalarmult(dp_window[w_start], -1*responses[w_start][0]));
			
			// rank-1 downdate mul1 (inverse of X'X^-1)
			
			denom = -1 + MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(dp_window[w_start]), mul1), dp_window[w_start])[0][0];
			
			nom1 = MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(mul1, dp_window[w_start]), MatrixOp.transpose(dp_window[w_start])),mul1);
			
			// compute params
			
			mul1 = MatrixOp.mat_add(mul1, MatrixOp.scalarmult(nom1, -1/denom));
			
			params = MatrixOp.mult(mul1, mul2);
			
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
		
		count_dps_in_window(); // this sets n;
		long squared_res_sum = 0;
		
		for(int ptr = w_start, ctr = 0; ctr < n; ptr = (ptr + 1) % w_size, ctr++)
			squared_res_sum = (long) Math.pow((responses[ptr][0] - MatrixOp.mult(MatrixOp.transpose(dp), params)[0][0])*10000, 2);
		
		running_residual_variance = squared_res_sum / 100000000.0*(n - feature_count);
		
	}
	
	public void print_params() {
		
		if(map2fs) {
			for(int ctr = 0; ctr < params.length; ctr++) {
				System.out.print(params[ctr][0] + " ");
				for(int ctr2 = 0; ctr2 < nlinmap.exp; ctr2++) {
					if(nlinmap.input_mapping[ctr][ctr2] > -1) System.out.print("x_"+nlinmap.input_mapping[ctr][ctr2]);
					else if(nlinmap.input_mapping[ctr][ctr2] > Integer.MIN_VALUE) System.out.print("log(x_"+(-1*nlinmap.input_mapping[ctr][ctr2]-1)+")");
				}
				System.out.println();
			}
		}
		else {
			for(int ctr = 0; ctr < params.length; ctr++)
				System.out.println(params[ctr][0] + " x_" + ctr);
		}
	}
}
