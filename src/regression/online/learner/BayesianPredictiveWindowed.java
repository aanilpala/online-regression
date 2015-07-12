package regression.online.learner;

import regression.online.model.Prediction;
import regression.online.util.MatrixOp;
import regression.online.util.MatrixPrinter;
import regression.online.util.NLInputMapper;

public class BayesianPredictiveWindowed extends WindowRegressor {
	
	double[][] mul1;
	double[][] mul2;   // column matrix
	
	double a; //measurement_precision;
	double b; //weight_precision;

	
	public BayesianPredictiveWindowed(int input_width, int window_size, double sigma_y, double sigma_w, int tuning_mode, boolean update_inhibator) {
		
		super(false, input_width, window_size, tuning_mode, update_inhibator);
		
		a = 1/(sigma_y*sigma_y);
		b = 1/(sigma_w*sigma_w);
	
		mul1 = new double[feature_count][feature_count];
		mul2 = new double[feature_count][1];
		
		for(int ctr = 0; ctr < feature_count; ctr++) {
			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
				if(ctr == ctr2) mul1[ctr][ctr2] = a/b;
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
		
//		System.out.println(update_count);
//		System.out.println("mul1:");
//		MatrixPrinter.print_matrix(mul1);
		
		double pp = MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(dp), mul1), mul2)[0][0];
		
		double pred_variance = MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(dp), MatrixOp.scalarmult(mul1, 1/a)), dp)[0][0]; 
		
		return new Prediction(pp, pred_variance);
		
	}
	
	public void update(double[][] dp, double y, Prediction prediction) throws Exception {
		
		int index = getIndexForDp(dp);
		
		if(index != -1) {
			// reject the update
			// avg the response for the duplicate point
			
			responses[index][0] = (y + responses[index][0])/2.0;
			return;
		}
						
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
			
			// update mul1
			
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
		
		update_count++;
		update_prediction_errors(y, prediction.point_prediction);
		
		if(is_tuning_time()) {
			double[][] weight_cov = get_weights_cov_matrix();
			
			// b approximation
//			double avg_var = 0;
//			for(int ctr = 0; ctr < weight_cov.length; ctr++)
//				avg_var += weight_cov[ctr][ctr];
//			
//			avg_var /= weight_cov.length;
//			
//			double old_b = b;
//			b = 1/avg_var;
			
			// recomputing mul1, mul2 and params
			
			double design_matrix[][] = new double[feature_count][w_size];  
			
			// computing the design matrix
			for(int ctr = 0; ctr < w_size; ctr++) {
				for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
					double entry = dp_window[(w_start + ctr) % w_size][ctr2][0];
					design_matrix[ctr2][(w_start + ctr) % w_size] = entry;
				}
			}
			
			// stupid stuff
			for (int ctr = 0; ctr < weight_cov.length; ctr++) {
				for (int ctr2 = 0; ctr2 < weight_cov.length; ctr2++)
					if(ctr != ctr2) weight_cov[ctr][ctr2] = 0;
			}
			
//			mul1 = MatrixOp.scalarmult(MatrixOp.fast_invert_psd(MatrixOp.mat_add(MatrixOp.scalarmult(MatrixOp.mult(design_matrix, MatrixOp.transpose(design_matrix)), a), MatrixOp.fast_invert_psd(weight_cov))), a);
			mul1 = MatrixOp.fast_invert_psd(MatrixOp.mat_add(MatrixOp.mult(design_matrix, MatrixOp.transpose(design_matrix)), MatrixOp.scalarmult(MatrixOp.fast_invert_psd(weight_cov), 1/a)));
			
			double[][] responses_vector = new double[w_size][1];
			
			// creating response vector
			for(int ctr = 0; ctr < w_size; ctr++) 
				responses_vector[ctr][0] = responses[(w_start + ctr) % w_size][0];
			
			mul2 = MatrixOp.mult(design_matrix, responses_vector);
		}
	}
	
}
