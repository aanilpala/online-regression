package regression.offline.learner;

import regression.online.model.Prediction;
import regression.util.MatrixOp;
import regression.util.NLInputMapper;

public abstract class BatchRegressor {

	public int feature_count;
	protected NLInputMapper nlinmap;
	public boolean map2fs;
	public String name;
	
	public int training_set_size;
	
	protected double[] input_scaler;
	protected double target_scaler;
	protected double[] input_means;
	protected double target_mean;
	
	protected boolean verbouse = false;
	
	public BatchRegressor(int input_width, int training_set_size, boolean map2fs) {
		
		this.map2fs = map2fs;
		this.training_set_size = training_set_size;
		
		name = this.getClass().getName() + (map2fs ? "Mapped" : "");
		name += training_set_size;
		
		name = name.split("\\.")[3];
		
		if(map2fs) {
			nlinmap = new NLInputMapper(input_width, 2, true, true);
			this.feature_count = nlinmap.feature_dim;
		}
		else {
			this.feature_count = input_width;
		}
		
		input_scaler = new double[input_width];
		input_means = new double[input_width];
		
		for (int ctr = 0; ctr < input_scaler.length; ctr++)
			input_scaler[ctr] = 1;
		
		target_scaler = 1;
		
		for (int ctr = 0; ctr < input_means.length; ctr++)
			input_means[ctr] = 0;
		
		target_mean = 0;
		
	}
	
	public void compute_means(double[][][] dps, double[][] targets) {
		
		long sum = 0;
		
		for(int ctr = 0; ctr < training_set_size; ctr++) {
			sum += targets[ctr][0]*1000000;
		}
		target_mean = (sum / 1000000.0) / training_set_size;
				
		for(int ctr = 0; ctr < input_means.length; ctr++) {
			sum = 0;
			for(int ctr2 = 0; ctr2 < training_set_size; ctr2++) {
				sum += dps[ctr2][ctr][0]*1000000;
			}
			input_means[ctr] = (sum / training_set_size) / 1000000.0;
		}
	}
	
	public void scale_training_data(double[][][] dps, double[][] targets) {
		
		for(int ctr = 0; ctr < training_set_size; ctr++) {
			targets[ctr][0] = target_prescaler(targets[ctr][0]);
		}
		
		for(int ctr = 0; ctr < training_set_size; ctr++) {
			dps[ctr] = scale_input(dps[ctr]);
		}
	}
	
	public double[][] scale_input(double[][] dp) {
		
		double[][] scaled_dp = new double[dp.length][1];
		
		for (int ctr = 0; ctr < input_scaler.length; ctr++)
			scaled_dp[ctr][0] = input_scaler[ctr]*dp[ctr][0];
		
		return scaled_dp;
	}
	
	public double[][] revert_input(double[][] scaled_dp) {
		
		double[][] dp = new double[scaled_dp.length][1];
		
		for (int ctr = 0; ctr < input_scaler.length; ctr++)
			dp[ctr][0] = (1/input_scaler[ctr])*scaled_dp[ctr][0];
		
		return dp;
	}
	
	public double target_prescaler(double target) {
		return target_scaler*target;
	}
	
	public double target_postscaler(double target) {
		return (1.0/target_scaler)*target;
	}
	
	public void determine_scaling_factors() {
		
		if(target_mean != 0)
			target_scaler = 1/(target_mean);
		
		for (int ctr = 0; ctr < input_means.length; ctr++) {
			if(input_means[ctr] == 0) continue;
			input_scaler[ctr] = 1/(input_means[ctr]);
		}	
	}
	
	public double predict(double[][] dp) throws Exception { return 0; };
	public void train(double[][][] dps, double[][] targets) throws Exception {}

	public int training_size() {
		return training_set_size;
	}

	public String get_name() {
		return name;  
	}
	
	protected double[][] get_weights_cov_matrix(double[][][] training_set) throws Exception {
		
		double[][] design_matrix = new double[training_set_size][feature_count];
		double[][] weight_cov = new double[training_set_size][feature_count];
		double[][] ones = new double[training_set_size][1];
		
		for(int ctr = 0; ctr < training_set_size; ctr++) {
			for(int ctr2 = 0; ctr2 < feature_count; ctr2++) {
				design_matrix[ctr][ctr2] = training_set[ctr][ctr2][0];
			}
		}
		
		for(int ctr = 0; ctr < training_set_size; ctr++)
			ones[ctr][0] = 1;
		
		design_matrix = MatrixOp.mat_subtract(design_matrix, MatrixOp.scalarmult(MatrixOp.mult(MatrixOp.mult(ones, MatrixOp.transpose(ones)), design_matrix), 1.0/training_set_size));
		weight_cov = MatrixOp.scalarmult(MatrixOp.mult(MatrixOp.transpose(design_matrix), design_matrix), 1.0/training_set_size);
		
		return weight_cov;
		
	}
	
	
}
