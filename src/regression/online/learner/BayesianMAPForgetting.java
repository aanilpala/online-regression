package regression.online.learner;

import java.io.IOException;

import regression.online.model.Prediction;
import regression.util.MatrixOp;
import regression.util.MatrixPrinter;

public class BayesianMAPForgetting extends OnlineRegressor {
	
	double[][][] v;
	double[][][] params; // column matrices
	
	double a; //measurement_precision;
	double b; //weight_precision;
	
	double forgetting_factor;
	
	public BayesianMAPForgetting(int input_width, double forgetting_factor, boolean map2fs, double signal_stddev, double weight_stddev) {
		
		super(map2fs, input_width);
		
		this.forgetting_factor = forgetting_factor;
		this.name += "FR" + forgetting_factor*100; 
		
		a = 1/(signal_stddev*signal_stddev);
		b = 1/(weight_stddev*weight_stddev);
		
		v = new double[3][feature_count][feature_count];
		params = new double[3][feature_count][1];
		
		for(int ctr = 0; ctr < feature_count; ctr++) {
			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
				for(int ctr3 = 0; ctr3 < 3; ctr3++) {
					if(ctr == ctr2) v[ctr3][ctr][ctr2] = b/a;
					else v[ctr3][ctr][ctr2] = 0;
				}
			}
		}
		
		for(int ctr = 0; ctr < feature_count; ctr++) {
			for(int ctr3 = 0; ctr3 < 3; ctr3++) {
				params[ctr3][ctr][0] = 0;
			}
		}
		
	}
	
	public Prediction predict(double[][] dp) throws Exception {
		
		if(map2fs) dp = nlinmap.map(dp);
		
		double pp = MatrixOp.mult(MatrixOp.transpose(dp), params[0])[0][0];
		double lp = MatrixOp.mult(MatrixOp.transpose(dp), params[1])[0][0];
		double up = MatrixOp.mult(MatrixOp.transpose(dp), params[2])[0][0];
		
		
		boolean interval_reversal = false;
		if(lp <= pp && pp <= up) interval_reversal = false;
		else interval_reversal = true;
			
//		System.out.print(interval_reversal ? "*" : "");
//		System.out.println(lp + " - " + pp + " - " + up);
		
		// correction
		if(interval_reversal) {
			if(lp > up) lp = up;
			if(lp > pp) up = lp;
			if(up < pp) up = pp;
		}
		
		return new Prediction(pp, up, lp, up, lp);
		
	}
	
	public void update(double[][] dp, double y, Prediction prediction) throws Exception {
		
		if(map2fs) dp = nlinmap.map(dp);
		
		int region;
		
		if(update_count < burn_in_count) region = -1;
		else if(y > prediction.upper_bound && y > prediction.point_prediction) region = 3;
		else if(y >= prediction.lower_bound &&  y <= prediction.upper_bound) region = 2;
		else if(y < prediction.lower_bound) region = 1;
		else {
			throw new Exception("UNDEFINED REGION");
		}
		
		// FOR TESTING
		
//		System.out.println(y);
//		System.out.println(prediction.toString());
//				
//		MatrixPrinter.print_matrix(v[0]);
//				
//		try {
//			System.in.read();
//		} catch (IOException e) {
//			e.printStackTrace();
//		}
				
		// FOR TESTING --END
		
		// update V
		
		double denom = 1 + MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(dp), v[0]), dp)[0][0];
		
		double[][] nom1 = MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(v[0], dp), MatrixOp.transpose(dp)), v[0]);
		
		// compute b
		
		v[0] = MatrixOp.scalarmult(MatrixOp.mat_add(v[0], MatrixOp.scalarmult(nom1, -1/denom)), 1/(1-forgetting_factor));
				
		params[0] = MatrixOp.mat_add(params[0], MatrixOp.scalarmult(MatrixOp.mult(v[0], dp), y - prediction.point_prediction)) ;
		
		
		if(region == -1 || region == 1 || region == 2) {
			// update V
			
			double denom_1 = 1 + MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(dp), v[1]), dp)[0][0];
			
			double[][] nom1_1 = MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(v[1], dp), MatrixOp.transpose(dp)), v[1]);
			
			// compute b
			
			v[1] = MatrixOp.scalarmult(MatrixOp.mat_add(v[1], MatrixOp.scalarmult(nom1_1, -1/denom_1)), 1/(1-forgetting_factor));
					
			params[1] = MatrixOp.mat_add(params[1], MatrixOp.scalarmult(MatrixOp.mult(v[1], dp), y - MatrixOp.mult(MatrixOp.transpose(dp), params[1])[0][0])) ;
		}
		
		if(region == -1 || region == 2 || region == 3) {
			// update V
			
			double denom_1 = 1 + MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(dp), v[2]), dp)[0][0];
						
			double[][] nom1_1 = MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(v[2], dp), MatrixOp.transpose(dp)), v[2]);
						
			// compute b
						
			v[2] = MatrixOp.scalarmult(MatrixOp.mat_add(v[2], MatrixOp.scalarmult(nom1_1, -1/denom_1)), 1/(1-forgetting_factor));
								
			params[2] = MatrixOp.mat_add(params[2], MatrixOp.scalarmult(MatrixOp.mult(v[2], dp), y - MatrixOp.mult(MatrixOp.transpose(dp), params[2])[0][0])) ;
		}
		
		update_count++;
		
	}
	
	public void print_params() {
		
		if(map2fs) {
			for(int ctr = 0; ctr < params[0].length; ctr++) {
				System.out.print(params[0][ctr][0] + " ");
				for(int ctr2 = 0; ctr2 < nlinmap.exp; ctr2++) {
					if(nlinmap.input_mapping[ctr][ctr2] > -1) System.out.print("x_"+nlinmap.input_mapping[ctr][ctr2]);
					else if(nlinmap.input_mapping[ctr][ctr2] > Integer.MIN_VALUE) System.out.print("log(x_"+(-1*nlinmap.input_mapping[ctr][ctr2]-1)+")");
				}
				System.out.println();
			}
		}
		else {
			for(int ctr = 0; ctr < params[0].length; ctr++)
				System.out.println(params[0][ctr][0] + " x_" + ctr);
		}
	}
}
