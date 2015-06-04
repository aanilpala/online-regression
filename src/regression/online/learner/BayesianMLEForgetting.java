package regression.online.learner;

import regression.online.model.Prediction;
import regression.online.util.MatrixOp;

public class BayesianMLEForgetting extends Regressor {
	
	double[][][] v;
	double[][][] params; // column matrices
	
	double forgetting_factor = 0.9;
	
	public BayesianMLEForgetting(int input_width, boolean map2fs) {
		
		super(map2fs, input_width);
	
		name = "BayesianMLEForgetting" + (map2fs ? "_MAPPED" : "");
		
		v = new double[3][feature_count][feature_count];
		params = new double[3][feature_count][1];
		
		for(int ctr = 0; ctr < feature_count; ctr++) {
			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
				for(int ctr3 = 0; ctr3 < 3; ctr3++) {
					if(ctr == ctr2) v[ctr3][ctr][ctr2] = 10000;
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
		double lb = MatrixOp.mult(MatrixOp.transpose(dp), params[1])[0][0];
		double ub = MatrixOp.mult(MatrixOp.transpose(dp), params[2])[0][0];
		
		if(lb == 0 && ub == 0) {
			lb = Double.POSITIVE_INFINITY;
			ub = Double.NEGATIVE_INFINITY;
		}
		
		return new Prediction(pp, ub, lb, ub, lb);
		
	}
	
	public void update(double[][] dp, double y, Prediction prediction) throws Exception {
		
		if(map2fs) dp = nlinmap.map(dp);
		
		boolean overestimated = false;
		boolean underestimated = false;
		
		if(y > prediction.upper_bound) underestimated = true;
		if(y < prediction.lower_bound) overestimated = true;
		
		// update V
		
		double denom = 1 + MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(dp), v[0]), dp)[0][0];
		
		double[][] nom1 = MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(v[0], dp), MatrixOp.transpose(dp)), v[0]);
		
		// compute b
		
		v[0] = MatrixOp.scalarmult(MatrixOp.mat_add(v[0], MatrixOp.scalarmult(nom1, -1/denom)), 1/forgetting_factor);
				
		params[0] = MatrixOp.mat_add(params[0], MatrixOp.scalarmult(MatrixOp.mult(v[0], dp), y - prediction.point_prediction)) ;
		
		
		if(overestimated) {
			// update V
			
			double denom_1 = 1 + MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(dp), v[1]), dp)[0][0];
			
			double[][] nom1_1 = MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(v[1], dp), MatrixOp.transpose(dp)), v[1]);
			
			// compute b
			
			v[1] = MatrixOp.scalarmult(MatrixOp.mat_add(v[1], MatrixOp.scalarmult(nom1_1, -1/denom_1)), 1/forgetting_factor);
					
			params[1] = MatrixOp.mat_add(params[1], MatrixOp.scalarmult(MatrixOp.mult(v[1], dp), y - MatrixOp.mult(MatrixOp.transpose(dp), params[1])[0][0])) ;
		}
		
		if(underestimated) {
			// update V
			
			double denom_1 = 1 + MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(dp), v[2]), dp)[0][0];
						
			double[][] nom1_1 = MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(v[2], dp), MatrixOp.transpose(dp)), v[2]);
						
			// compute b
						
			v[2] = MatrixOp.scalarmult(MatrixOp.mat_add(v[2], MatrixOp.scalarmult(nom1_1, -1/denom_1)), 1/forgetting_factor);
								
			params[2] = MatrixOp.mat_add(params[2], MatrixOp.scalarmult(MatrixOp.mult(v[2], dp), y - MatrixOp.mult(MatrixOp.transpose(dp), params[2])[0][0])) ;
		}
		
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
