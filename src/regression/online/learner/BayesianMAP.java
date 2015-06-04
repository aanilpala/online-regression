package regression.online.learner;

import regression.online.model.Prediction;
import regression.online.util.MatrixOp;
import regression.online.util.NLInputMapper;

public class BayesianMAP extends Regressor {
	
	double[][][] a_inv;
	double[][][] v;	// column matrices
	double[][][] params; // column matrices
	
	double a; //measurement_precision;
	double b; //weight_precision;
	
	public BayesianMAP(int input_width, boolean map2fs, double signal_stddev, double weight_stddev) {
		
		super(map2fs, input_width);
		
		name = "BayesianMAP" + (map2fs ? "_MAPPED" : "");
		
		a = 1/(signal_stddev*signal_stddev);
		b = 1/(weight_stddev*weight_stddev);
	
		a_inv = new double[3][feature_count][feature_count];
		v = new double[3][feature_count][1];
		params = new double[3][feature_count][1];
		
		for(int ctr = 0; ctr < feature_count; ctr++) {
			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
				for(int ctr3 = 0; ctr3 < 3; ctr3++) {
					if(ctr == ctr2) a_inv[ctr3][ctr][ctr2] = b/a;
					else a_inv[ctr3][ctr][ctr2] = 0;
				}
			}
		}
		
		for(int ctr = 0; ctr < feature_count; ctr++) {
			for(int ctr3 = 0; ctr3 < 3; ctr3++) {
				v[ctr3][ctr][0] = 0;
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
		
		return new Prediction(pp, ub, lb);
		
	}
	
	public void update(double[][] dp, double y, Prediction prediction) throws Exception {
		
		if(map2fs) dp = nlinmap.map(dp);
		
		boolean overestimated = false;
		boolean underestimated = false;
		
		if(y > prediction.upper_bound) underestimated = true;
		if(y < prediction.lower_bound) overestimated = true;
		
		// update v
		
		v[0] = MatrixOp.mat_add(v[0], MatrixOp.scalarmult(dp, y));
		
		// update A^-1
		
		double denom = 1 + MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(dp), a_inv[0]), dp)[0][0];
		
		double[][] nom1 = MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(a_inv[0], dp), MatrixOp.transpose(dp)),a_inv[0]);
		
		// compute b
		
		a_inv[0] = MatrixOp.mat_add(a_inv[0], MatrixOp.scalarmult(nom1, -1/denom));
		
		params[0] = MatrixOp.mult(a_inv[0], v[0]);
		
		
		if(overestimated) {
			// update v

			v[1] = MatrixOp.mat_add(v[1], MatrixOp.scalarmult(dp, y));
			double denom_1 = 1 + MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(dp), a_inv[1]), dp)[0][0];
			double[][] nom1_1 = MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(a_inv[1], dp), MatrixOp.transpose(dp)),a_inv[1]);
			
			// compute b
			
			a_inv[1] = MatrixOp.mat_add(a_inv[1], MatrixOp.scalarmult(nom1_1, -1/denom_1));
			
			params[1] = MatrixOp.mult(a_inv[1], v[1]);
		}
		
		if(underestimated) {
			// update v
			
			v[2] = MatrixOp.mat_add(v[2], MatrixOp.scalarmult(dp, y));
			double denom_2 = 1 + MatrixOp.mult(MatrixOp.mult(MatrixOp.transpose(dp), a_inv[2]), dp)[0][0];
			double[][] nom1_2 = MatrixOp.mult(MatrixOp.mult(MatrixOp.mult(a_inv[2], dp), MatrixOp.transpose(dp)),a_inv[2]);
			
			// compute b
			
			a_inv[2] = MatrixOp.mat_add(a_inv[2], MatrixOp.scalarmult(nom1_2, -1/denom_2));
			
			params[2] = MatrixOp.mult(a_inv[2], v[2]);
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
