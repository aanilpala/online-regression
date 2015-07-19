package regression.offline.learner;

import regression.online.model.Prediction;
import regression.util.MatrixOp;

public class BayesianMLEBatch extends BatchRegressor {

	double[][] params; // column matrices 
	
	public BayesianMLEBatch(int input_width, int training_set_size, boolean map2fs) {
		super(input_width, training_set_size, map2fs);
		
		params = new double[feature_count][1];
	}
	
	@Override
	public void train(double[][][] dps, double[][] targets) throws Exception {
		
		compute_means(dps, targets);
		determine_scaling_factors();
		scale_training_data(dps, targets);
		
		double[][] design_matrix;
		design_matrix = new double[feature_count][training_set_size];
		
		for(int ctr = 0; ctr < training_set_size; ctr++) {
			
			double[][] cur_dp = dps[ctr];
			if(map2fs) cur_dp = nlinmap.map(cur_dp);
			
			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
				double entry = cur_dp[ctr2][0];
				design_matrix[ctr2][ctr] = entry;
			}
		}
		
		double[][] mul1 = MatrixOp.fast_invert_psd(MatrixOp.mult(design_matrix, MatrixOp.transpose(design_matrix)));
		double[][] mul2 = MatrixOp.mult(design_matrix, targets);
		
		params = MatrixOp.mult(mul1, mul2);
	}

	@Override
	public double predict(double[][] dp) throws Exception {
		
		dp = scale_input(dp);
		if(map2fs) dp = nlinmap.map(dp);
		
		double pp = MatrixOp.mult(MatrixOp.transpose(dp), params)[0][0];
		return target_postscaler(pp);
	}
	
}
